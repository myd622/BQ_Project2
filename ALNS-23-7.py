import numpy as np
import pandas as pd
import random
import copy
import time
# import matplotlib

# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt
from collections import defaultdict
import math
import networkx as nx
from collections import deque
import functools  # 添加导入语句


# 设置随机种子保证结果可复现
# random.seed(42)
# np.random.seed(42)


class ProblemInstance:
    """问题实例类，包含所有输入数据"""

    def __init__(self, ships, berths, Bj, Q, tvc, vj, M):
        """
        初始化问题实例

        参数:
        ships -- 船舶列表，每个船舶是字典，包含:
                'vessel': 船舶编号
                'tva': 预计到达时间
                'tvd': 预计离泊时间
                'tasks': 任务列表，每个任务是元组 (任务类型, 作业量)
        berths -- 泊位字典，格式: {'泊位名': [支持的任务类型列表]}
        Bj -- 任务到泊位的映射字典，格式: {'任务类型': [支持的泊位列表]}
        Q -- 岸桥列表
        tvc -- 移泊时间（小时）
        vj -- 任务作业速率字典，格式: {'任务类型': 速率}
        M -- 大M值（用于线性规划约束)
        """
        self.ships = ships
        self.berths = berths
        self.Bj = Bj
        self.Q = Q
        self.tvc = tvc
        self.vj = vj
        self.M = M

        # 互斥任务集合（加油与其他任务互斥）
        self.JE = ['j1']  # Refueling是互斥任务

        # 需要岸桥的任务集合（卸货和装货）
        self.JQ = ['j2', 'j3']  # Unloading and Loading

        # 构建船舶到索引的映射
        self.vessel_to_index = {ship['vessel']: idx for idx, ship in enumerate(ships)}

        # 构建任务优先级图
        self.priority_edges = {}
        for ship in ships:
            vessel = ship['vessel']
            self.priority_edges[vessel] = self.build_priority_edges(ship['tasks'])

    def build_priority_edges(self, tasks):
        """构建任务优先级边集合（不是DAG，而是成对约束）"""
        edges = []
        task_types = [task[0] for task in tasks]

        # 添加优先级约束: 卸载(j2)必须在装载(j3)之前
        if 'j2' in task_types and 'j3' in task_types:
            edges.append(('j2', 'j3'))

        return edges

    def get_priority_constraints(self, vessel):
        """获取任务的优先级约束列表"""
        return self.priority_edges.get(vessel, [])

    def get_vessel_tasks(self, vessel_no):
        """获取指定船舶的所有任务"""
        idx = self.vessel_to_index[vessel_no]
        return self.ships[idx]['tasks']

    def get_task_duration(self, task_type):
        """获取任务的标准持续时间（作业量/作业速率）"""
        return self.vj[task_type]

    def get_berths_for_task(self, task_type):
        """获取支持该任务类型的泊位列表"""
        return self.Bj[task_type]

    def is_mutually_exclusive(self, task_type):
        """检查任务是否是互斥任务"""
        return task_type in self.JE

    def requires_quay_crane(self, task_type):
        """检查任务是否需要岸桥"""
        return task_type in self.JQ


class Solution:
    """解表示类，包含调度方案的所有信息"""

    def __init__(self, instance):
        """
        初始化解表示

        参数:
        instance -- ProblemInstance对象
        """
        self.instance = instance
        self.cost = float('inf')  # 目标函数值（总延迟）

        # 决策变量
        self.X = {}  # 泊位分配: (vessel, task, berth) -> 0/1
        self.Y = {}  # 岸桥分配: (vessel, task, crane) -> 0/1
        self.F = {}  # 泊位顺序: (vessel, vessel_prime, berth) -> 0/1
        self.G = {}  # 岸桥任务顺序: (vessel, vessel_prime, crane, task, task_prime) -> 0/1
        self.O = {}  # 移泊顺序: (vessel, berth, berth_prime) -> 0/1
        self.Z = {}  # 船舶使用泊位: (vessel, berth) -> 0/1

        # 时间变量
        self.task_start = {}  # 任务开始时间: (vessel, task) -> float
        self.task_end = {}  # 任务结束时间: (vessel, task) -> float
        self.vessel_end = {}  # 船舶结束时间: vessel -> float
        self.berth_occupation_start = {}  # 船舶在泊位开始时间: (vessel, berth) -> float
        self.berth_occupation_end = {}  # 船舶在泊位结束时间: (vessel, berth) -> float
        self.delay = {}  # 船舶延迟时间: vessel -> float

        # 序列变量
        self.berth_sequences = {berth: [] for berth in instance.berths}  # 泊位占用序列
        self.crane_sequences = {crane: [] for crane in instance.Q}  # 岸桥占用序列
        self.vessel_movements = {}  # 船舶移泊序列: vessel -> [berth1, berth2, ...]

        # 辅助数据结构
        self.task_to_berth = {}  # (vessel, task) -> berth
        self.task_to_crane = {}  # (vessel, task) -> crane (仅对需要岸桥的任务)

        # 初始化所有决策变量为0
        self.initialize_decision_variables()

    def initialize_decision_variables(self):
        """初始化所有决策变量为0"""
        instance = self.instance

        # 初始化泊位分配X
        for ship in instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                for berth in instance.Bj[task_type]:
                    self.X[(vessel, task_type, berth)] = 0

        # 初始化岸桥分配Y
        for ship in instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                if task_type in instance.JQ:
                    for crane in instance.Q:
                        self.Y[(vessel, task_type, crane)] = 0

        # 初始化泊位顺序F
        for vessel1 in [s['vessel'] for s in instance.ships]:
            for vessel2 in [s['vessel'] for s in instance.ships]:
                if vessel1 != vessel2:
                    for berth in instance.berths:
                        self.F[(vessel1, vessel2, berth)] = 0

        # 初始化岸桥任务顺序G
        for vessel1 in [s['vessel'] for s in instance.ships]:
            for vessel2 in [s['vessel'] for s in instance.ships]:
                if vessel1 != vessel2:
                    for crane in instance.Q:
                        for task1 in ['j2', 'j3']:  # 只考虑需要岸桥的任务
                            for task2 in ['j2', 'j3']:
                                self.G[(vessel1, vessel2, crane, task1, task2)] = 0

        # 初始化移泊顺序O
        for ship in instance.ships:
            vessel = ship['vessel']
            for berth1 in instance.berths:
                for berth2 in instance.berths:
                    if berth1 != berth2:
                        self.O[(vessel, berth1, berth2)] = 0

        # 初始化船舶使用泊位Z
        for ship in instance.ships:
            vessel = ship['vessel']
            for berth in instance.berths:
                self.Z[(vessel, berth)] = 0

    def calculate_cost(self):
        """计算解的总成本（总延迟）"""
        total_delay = 0.0
        for ship in self.instance.ships:
            vessel = ship['vessel']
            if vessel in self.delay:
                total_delay += self.delay[vessel]
        return total_delay

    def is_feasible(self):
        """检查解是否满足所有约束条件，并输出详细的错误信息"""
        instance = self.instance
        errors = []  # 存储所有错误信息

        # 1. 检查泊位分配约束
        for ship in instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                # 每个任务必须分配到一个泊位
                assigned = False
                for berth in instance.berths:
                    if self.X.get((vessel, task_type, berth), 0) == 1:
                        assigned = True
                        break
                if not assigned:
                    errors.append(f"错误: 船舶 {vessel} 的任务 {task_type} 未分配到泊位")

        # 2. 检查岸桥分配约束
        for ship in instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                if task_type in instance.JQ:
                    # 需要岸桥的任务必须分配到一个岸桥
                    assigned = False
                    for crane in instance.Q:
                        if self.Y.get((vessel, task_type, crane), 0) == 1:
                            assigned = True
                            break
                    if not assigned:
                        errors.append(f"错误: 船舶 {vessel} 的任务 {task_type} 未分配到岸桥")

        # 3. 检查互斥约束
        for ship in instance.ships:
            vessel = ship['vessel']
            for berth in instance.berths:
                exclusive_task_found = False
                exclusive_task_type = None
                for task in ship['tasks']:
                    task_type = task[0]
                    if self.X.get((vessel, task_type, berth), 0) == 1:
                        if task_type in instance.JE:
                            if exclusive_task_found:
                                errors.append(f"错误: 船舶 {vessel} 在泊位 {berth} 上有多个互斥任务")
                            exclusive_task_found = True
                            exclusive_task_type = task_type

                # 如果有互斥任务，泊位上不能有其他任务
                if exclusive_task_found:
                    for other_task in ship['tasks']:
                        other_task_type = other_task[0]
                        if other_task_type != exclusive_task_type and self.X.get((vessel, other_task_type, berth),
                                                                                 0) == 1:
                            errors.append(
                                f"错误: 船舶 {vessel} 在泊位 {berth} 上互斥任务 {exclusive_task_type} 与其他任务 {other_task_type} 冲突")

        # 4. 检查优先级约束
        for ship in instance.ships:
            vessel = ship['vessel']
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                # 确保优先任务在当前任务之前完成
                if (vessel, j_prime) in self.task_end and (vessel, j) in self.task_start:
                    if self.task_end[(vessel, j_prime)] > self.task_start[(vessel, j)]:
                        errors.append(
                            f"错误: 船舶 {vessel} 任务 {j_prime} 结束时间 {self.task_end[(vessel, j_prime)]} > 任务 {j} 开始时间 {self.task_start[(vessel, j)]}")

        # 5. 检查泊位时间约束
        for berth in instance.berths:
            occupations = self.berth_sequences.get(berth, [])
            # 按开始时间排序
            occupations.sort(key=lambda x: x['start'])
            last_end = -1
            for occ in occupations:
                if occ['start'] < last_end:
                    errors.append(f"错误: 泊位 {berth} 上船舶 {occ['vessel']} 开始时间 {occ['start']} < 前一个结束时间 {last_end}")
                last_end = occ['end']

        # 6. 检查岸桥时间约束
        for crane in instance.Q:
            crane_tasks = []
            for ship in instance.ships:
                vessel = ship['vessel']
                for task in ship['tasks']:
                    task_type = task[0]
                    if task_type in instance.JQ:
                        if self.Y.get((vessel, task_type, crane), 0) == 1:
                            crane_tasks.append({
                                'vessel': vessel,
                                'task': task_type,
                                'start': self.task_start.get((vessel, task_type), 0),
                                'end': self.task_end.get((vessel, task_type), 0)
                            })
            # 按开始时间排序
            crane_tasks.sort(key=lambda x: x['start'])
            last_end = -1
            for task in crane_tasks:
                if task['start'] < last_end:
                    errors.append(f"错误: 岸桥 {crane} 上任务 {task['task']} 开始时间 {task['start']} < 前一个结束时间 {last_end}")
                last_end = task['end']

        # 7. 检查移泊约束
        for vessel in self.vessel_movements:
            berth_sequence = self.vessel_movements[vessel]
            if len(berth_sequence) > 1:
                for i in range(len(berth_sequence) - 1):
                    current_berth = berth_sequence[i]
                    next_berth = berth_sequence[i + 1]

                    # 获取当前泊位的结束时间
                    current_end = self.berth_occupation_end.get((vessel, current_berth), 0)

                    # 获取下一个泊位的开始时间
                    next_start = self.berth_occupation_start.get((vessel, next_berth), 0)

                    # 检查移泊时间约束：下一个泊位的开始时间 >= 当前泊位的结束时间 + 移泊时间
                    if next_start < current_end + instance.tvc:
                        errors.append(f"错误: 船舶 {vessel} 从泊位 {current_berth} 到 {next_berth} 的移泊时间不满足约束")
                        errors.append(f"     当前泊位结束时间: {current_end}, 下一个泊位开始时间: {next_start}, 移泊时间: {instance.tvc}")

        # 8. 检查任务时间约束
        for ship in instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                if (vessel, task_type) in self.task_start and (vessel, task_type) in self.task_end:
                    # 检查任务开始时间不早于船舶到达时间
                    if self.task_start[(vessel, task_type)] < ship['tva']:
                        errors.append(
                            f"错误: 船舶 {vessel} 任务 {task_type} 开始时间 {self.task_start[(vessel, task_type)]} < 船舶到达时间 {ship['tva']}")

                    # 检查任务结束时间计算是否正确
                    expected_duration = task[1] / instance.vj[task_type]
                    actual_duration = self.task_end[(vessel, task_type)] - self.task_start[(vessel, task_type)]
                    if abs(actual_duration - expected_duration) > 1e-6:
                        errors.append(f"错误: 船舶 {vessel} 任务 {task_type} 持续时间不正确")
                        errors.append(f"     预期: {expected_duration}, 实际: {actual_duration}")

        # 输出所有错误
        if errors:
            print("解不可行，发现以下错误:")
            for error in errors:
                print(f"  {error}")
            return False

        return True


class ALNSSolver:
    """自适应大邻域搜索算法求解器"""

    def __init__(self, instance, max_iter=50, max_time=3600, no_improve_max=50,
                 initial_temp=1000, cooling_rate=0.99, min_temp=0.1,
                 destroy_operators=None, repair_operators=None):
        """
        初始化ALNS求解器

        参数:
        instance -- ProblemInstance对象
        max_iter -- 最大迭代次数
        max_time -- 最大运行时间（秒）
        no_improve_max -- 无改进最大迭代次数
        initial_temp -- 初始温度（模拟退火）
        cooling_rate -- 冷却率（模拟退火）
        min_temp -- 最低温度（模拟退火）
        destroy_operators -- 破坏算子列表
        repair_operators -- 修复算子列表
        """
        self.instance = instance
        self.max_iter = max_iter
        self.max_time = max_time
        self.no_improve_max = no_improve_max
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

        # 破坏算子
        self.destroy_operators = destroy_operators or [
            self.random_removal_single,
            self.random_removal_three,
            self.random_removal_vessel,
            self.delay_based_removal_single,
            self.delay_based_removal_three,
            self.delay_based_removal_vessel
        ]

        # 修复算子
        self.repair_operators = repair_operators or [
            self.random_insertion,
            self.greedy_insertion,
            self.regret_insertion
        ]

        # 算子管理器
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.destroy_scores = [0] * len(self.destroy_operators)
        self.destroy_counts = [0] * len(self.destroy_operators)

        self.repair_weights = [1.0] * len(self.repair_operators)
        self.repair_scores = [0] * len(self.repair_operators)
        self.repair_counts = [0] * len(self.repair_operators)

        # 得分参数
        self.sigma1 = 1.0  # 改进解得分
        self.sigma2 = 0.5  # 接受但未改进解得分
        self.sigma3 = 0.1  # 拒绝解得分
        self.lambda_ = 0.8  # 遗忘因子
        self.update_freq = 100  # 权重更新频率

        # 多样性保持
        self.diversity_freq = 50  # 多样性检查频率
        self.perturb_freq = 20  # 扰动频率

        # 记录
        self.best_solution = None
        self.current_solution = None
        self.history = []  # 记录每次迭代的成本
        self.destroy_names = ["DR1", "DR2", "DR3", "DR4", "DR5", "DR6"]
        self.repair_names = ["RR1", "RR2", "RR3"]

    def construct_initial_solution(self):
        """构造初始可行解（考虑并行任务执行和移泊约束）"""
        # 按预计到达时间排序船舶
        sorted_ships = sorted(self.instance.ships, key=lambda x: x['tva'])

        # 创建新解
        solution = Solution(self.instance)

        # 处理每艘船舶
        for ship in sorted_ships:
            vessel = ship['vessel']
            tva = ship['tva']
            tvd = ship['tvd']
            tasks = ship['tasks']

            # 步骤1: 分配泊位，满足互斥约束
            # 识别互斥任务
            mutually_exclusive_tasks = [task for task in tasks if task[0] in self.instance.JE]
            non_mutually_exclusive_tasks = [task for task in tasks if task[0] not in self.instance.JE]

            # 任务分组（按泊位）
            berth_tasks = defaultdict(list)  # berth -> [task]

            # 分配互斥任务到独占泊位
            for task in mutually_exclusive_tasks:
                task_type, workload = task
                possible_berths = self.instance.Bj[task_type]

                # 根据各个泊位的最后一个船舶占用的结束时间对possible_berths进行排序（升序）
                # 获取每个泊位的最后一个船舶占用的结束时间
                berth_end_times = {}
                for berth in possible_berths:
                    if solution.berth_sequences[berth]:
                        last_occupation = solution.berth_sequences[berth][-1]
                        berth_end_times[berth] = last_occupation['end']
                    else:
                        berth_end_times[berth] = 0

                sort_possible_berths = sorted(possible_berths, key=lambda b: berth_end_times[b])

                best_berth = None
                for berth in sort_possible_berths:
                    # 检查泊位是否已被该船舶使用（互斥任务需要独占泊位）
                    if solution.Z.get((vessel, berth), 0) == 1:
                        continue
                    else:
                        best_berth = berth
                        break

                if best_berth is None:
                    # 如果没有可用泊位，选择第一个泊位
                    best_berth = sort_possible_berths[0] if sort_possible_berths else possible_berths[0]

                # 分配泊位
                solution.X[(vessel, task_type, best_berth)] = 1
                solution.task_to_berth[(vessel, task_type)] = best_berth
                solution.Z[(vessel, best_berth)] = 1

                # 添加到泊位任务组
                berth_tasks[best_berth].append(task)

            # 分配非互斥任务
            for task in non_mutually_exclusive_tasks:
                task_type, workload = task
                possible_berths = self.instance.Bj[task_type]

                # 选择最早可用的泊位
                best_berth = None

                # 根据各个泊位的最后一个船舶占用的结束时间对possible_berths进行排序（升序）
                valid_berths = []
                for berth in possible_berths:
                    # 检查互斥约束：泊位上不能有互斥任务
                    has_exclusive = False
                    for t in berth_tasks[berth]:
                        if t[0] in self.instance.JE:
                            has_exclusive = True
                            break
                    if not has_exclusive:
                        valid_berths.append(berth)

                # 获取每个泊位的最后一个船舶占用的结束时间
                berth_end_times = {}
                for berth in valid_berths:
                    if solution.berth_sequences[berth]:
                        last_occupation = solution.berth_sequences[berth][-1]
                        berth_end_times[berth] = last_occupation['end']
                    else:
                        berth_end_times[berth] = 0

                sort_valid_berths = sorted(valid_berths, key=lambda b: berth_end_times[b])

                best_berth = sort_valid_berths[0]

                '''
                for berth in sort_valid_berths:
                    # 检查泊位是否已被该船舶使用
                    if solution.Z.get((vessel, berth), 0) == 1:
                        continue
                    else:
                        best_berth = berth
                        break

                if best_berth is None and sort_valid_berths:
                    best_berth = sort_valid_berths[0]
                elif not sort_valid_berths:
                    continue
                '''

                # 分配泊位
                solution.X[(vessel, task_type, best_berth)] = 1
                solution.task_to_berth[(vessel, task_type)] = best_berth
                solution.Z[(vessel, best_berth)] = 1

                # 添加到泊位任务组
                berth_tasks[best_berth].append(task)

            # 步骤2: 确定泊位访问序列（考虑移泊约束和任务优先级）
            # used_berths = list(berth_tasks.keys())
            used_berths = [berth for berth in berth_tasks.keys() if berth_tasks[berth]]

            # 如果没有使用泊位，跳过
            if not used_berths:
                continue

            # 根据任务优先级约束对泊位进行排序
            priority_graph = nx.DiGraph()
            priority_graph.add_nodes_from(used_berths)

            # 获取该船舶的优先级约束
            priority_edges = self.instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                # 获取这两个任务分配的泊位
                berth_prime = solution.task_to_berth.get((vessel, j_prime))
                berth_j = solution.task_to_berth.get((vessel, j))

                # 如果两个泊位不同，并且都不为空，则添加一条边从berth_prime到berth_j
                if berth_prime is not None and berth_j is not None and berth_prime != berth_j:
                    priority_graph.add_edge(berth_prime, berth_j)

            try:
                # 拓扑排序
                sorted_berths = list(nx.topological_sort(priority_graph))
            except nx.NetworkXUnfeasible:
                # 有环，则按任意顺序（例如按泊位名称排序）
                sorted_berths = sorted(used_berths)
                print(f"警告: 船舶 {vessel} 的泊位优先级图中存在环，使用任意顺序")

            solution.vessel_movements[vessel] = sorted_berths

            # 步骤3: 调度每个泊位上的任务组（考虑移泊约束）
            occupation_end_times = {}
            prev_berth_end = tva  # 船舶到达时间

            for i, berth in enumerate(sorted_berths):
                tasks_in_berth = berth_tasks[berth]

                # 调用函数1：计算当前循环下的船舶在当前循环下的泊位的占用时长
                TS = self.calculate_occupation_duration(vessel, berth, tasks_in_berth, solution)

                # 调用函数2：船舶v任务在泊位b占用调度
                occupation_start, occupation_end = self.schedule_berth_occupation_with_check(
                    solution, vessel, berth, tasks_in_berth, prev_berth_end, i, TS
                )

                prev_berth_end = occupation_end

                # 保存占用信息
                solution.berth_occupation_start[(vessel, berth)] = occupation_start
                solution.berth_occupation_end[(vessel, berth)] = occupation_end

                # 添加到泊位序列
                solution.berth_sequences[berth].append({
                    'vessel': vessel,
                    'start': occupation_start,
                    'end': occupation_end,
                    'tasks': tasks_in_berth
                })

                occupation_end_times[berth] = occupation_end
                solution.berth_sequences[berth].sort(key=lambda x: x['start'])

            # 步骤4: 计算船舶结束时间
            vessel_end = max(occupation_end_times.values()) if occupation_end_times else tva
            solution.vessel_end[vessel] = vessel_end
            solution.delay[vessel] = max(0, vessel_end - tvd)

            # 验证解可行性
            # if not solution.is_feasible():
            # print(ship)
            # print("警告: 初始解不可行")

        # 计算总成本
        solution.cost = solution.calculate_cost()

        # 验证解可行性
        if not solution.is_feasible():
            print("警告: 初始解不可行")

        # 输出解的所有详细信息
        print_solution_details(solution)

        return solution

    def calculate_occupation_duration(self, vessel, berth, tasks, solution):
        """计算船舶v在泊位b上的占用时长TS"""
        instance = self.instance

        # 获取该船舶的优先级约束
        priority_edges = instance.get_priority_constraints(vessel)

        # 分离有优先级关系的任务和无优先级关系的任务
        priority_tasks = set()
        for (j_prime, j) in priority_edges:
            priority_tasks.add(j_prime)
            priority_tasks.add(j)

        priority_task_list = []
        non_priority_task_list = []

        for task_type, workload in tasks:
            if task_type in priority_tasks:
                priority_task_list.append((task_type, workload))
            else:
                non_priority_task_list.append((task_type, workload))

        # 计算优先级任务的总时长
        priority_sum = 0
        for task_type, workload in priority_task_list:
            duration = workload / instance.vj[task_type]
            priority_sum += duration

        # 计算非优先级任务的最大时长
        non_priority_max = 0
        for task_type, workload in non_priority_task_list:
            duration = workload / instance.vj[task_type]
            if duration > non_priority_max:
                non_priority_max = duration

        # 计算所有任务的最大时长
        all_tasks_max = 0
        for task_type, workload in tasks:
            duration = workload / instance.vj[task_type]
            if duration > all_tasks_max:
                all_tasks_max = duration

        # 占用时长TS = max{优先级任务总时长, 非优先级任务最大时长, 所有任务最大时长}
        return max(priority_sum, non_priority_max, all_tasks_max)

    def schedule_berth_occupation_with_check(self, solution, vessel, berth, tasks, prev_berth_end, berth_index, TS):
        """船舶v任务在泊位b占用调度函数"""
        instance = self.instance
        # 步骤1：确定时间阈值TTT

        # 获取船舶的预计到达时间
        ship_data = next((s for s in instance.ships if s['vessel'] == vessel), None)
        if ship_data:
            tva = ship_data['tva']
        else:
            tva = 0

        # 步骤1：确定时间阈值TTT
        if berth_index == 0:
            # TTT = max(tva, solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'])
            TTT = tva
        else:
            TTT = max(tva, prev_berth_end + instance.tvc)

        '''
        # 步骤1：确定时间阈值TTT
        if berth_index == 0:
            TTT = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
        else:
            TTT = prev_berth_end + instance.tvc
        '''

        # 步骤2：对当前船舶v在当前泊位b任务进行调度
        berth_occupations = solution.berth_sequences[berth]

        # 检查泊位b是否有船舶占用
        if not berth_occupations:
            # 没有船舶占用
            start_time = TTT
            end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)
            return start_time, end_time
        else:
            # 有船舶占用
            # a) 检查第一个占用之前的时间段
            first_occupation = berth_occupations[0]
            if TTT + TS <= first_occupation['start']:
                start_time = TTT
                end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)
                if end_time <= first_occupation['start']:
                    return start_time, end_time
                else:
                    # 复原相关变量
                    self.rollback_scheduling(solution, vessel, berth, tasks)

            '''

            # b) 循环检查中间时间段
            for i in range(len(berth_occupations) - 1):
                current_occupation = berth_occupations[i]
                next_occupation = berth_occupations[i + 1]

                if current_occupation['end'] <= TTT:
                    # TTR = max(TTT, current_occupation['end'])
                    if next_occupation['start'] >= TTT + TS:
                        start_time = TTT
                        end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)
                        if end_time <= next_occupation['start']:
                            return start_time, end_time
                        else:
                            # 复原相关变量
                            self.rollback_scheduling(solution, vessel, berth, tasks)
            '''

            # b) 循环检查中间时间段
            for i in range(len(berth_occupations) - 1):
                current_occupation = berth_occupations[i]
                next_occupation = berth_occupations[i + 1]

                # 修改点：计算最早开始时间，考虑当前占用的结束时间
                earliest_start = max(TTT, current_occupation['end'])
                if earliest_start + TS <= next_occupation['start']:
                    start_time = earliest_start
                    end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)
                    if end_time <= next_occupation['start']:
                        return start_time, end_time
                    else:
                        self.rollback_scheduling(solution, vessel, berth, tasks)

            # c) 检查最后一个占用之后的时间段
            last_occupation = berth_occupations[-1]
            TTP = max(TTT, last_occupation['end'])
            start_time = TTP
            end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)
            return start_time, end_time

    def rollback_scheduling(self, solution, vessel, berth, tasks):
        """复原调度相关的决策变量"""
        instance = self.instance

        for task_type, workload in tasks:
            # 删除泊位分配
            solution.X[(vessel, task_type, berth)] = 0
            if (vessel, task_type) in solution.task_to_berth:
                del solution.task_to_berth[(vessel, task_type)]

            # 如果任务需要岸桥，删除岸桥分配
            if task_type in instance.JQ and (vessel, task_type) in solution.task_to_crane:
                crane = solution.task_to_crane[(vessel, task_type)]
                solution.Y[(vessel, task_type, crane)] = 0

                # 从岸桥序列中删除任务
                crane_sequence = solution.crane_sequences[crane]
                for i, task_info in enumerate(crane_sequence):
                    if task_info['vessel'] == vessel and task_info['task'] == task_type:
                        del crane_sequence[i]
                        break

                del solution.task_to_crane[(vessel, task_type)]

            # 删除时间变量
            if (vessel, task_type) in solution.task_start:
                del solution.task_start[(vessel, task_type)]
            if (vessel, task_type) in solution.task_end:
                del solution.task_end[(vessel, task_type)]

    def schedule_berth_occupation(self, vessel, berth, occupation_tasks, solution, start_time):
        """调度泊位占用内的任务（支持并行执行）"""
        instance = self.instance

        # 获取船舶的预计到达时间
        ship_data = next((s for s in instance.ships if s['vessel'] == vessel), None)
        if ship_data:
            tva = ship_data['tva']
        else:
            tva = 0

        # 确保开始时间不早于船舶到达时间
        start_time = max(start_time, tva)

        # 获取任务优先级约束
        priority_edges = instance.get_priority_constraints(vessel)

        # 创建任务依赖图（考虑优先级约束）
        occupation_task_types = {task[0] for task in occupation_tasks}
        task_graph = nx.DiGraph()
        for task in occupation_tasks:
            task_type = task[0]
            task_graph.add_node(task_type)

        # 添加优先级约束（仅适用于当前占用中的任务）
        for (j_prime, j) in priority_edges:
            # 只添加两个任务都在当前占用中的约束
            if any(t[0] == j_prime for t in occupation_tasks) and any(t[0] == j for t in occupation_tasks):
                task_graph.add_edge(j_prime, j)

        # 检查是否有环（理论上不应该有，但安全起见）
        if not nx.is_directed_acyclic_graph(task_graph):
            # 处理循环依赖：移除导致循环的边
            while not nx.is_directed_acyclic_graph(task_graph):
                # 找到并移除一个循环
                try:
                    cycle = nx.find_cycle(task_graph)
                    # 移除循环中的第一条边
                    task_graph.remove_edge(cycle[0][0], cycle[0][1])
                    print(f"警告: 船舶 {vessel} 在泊位 {berth} 的任务优先级图中存在环，移除了边 {cycle[0]}")
                except nx.NetworkXNoCycle:
                    break

        # 拓扑排序
        try:
            topological_order = list(nx.topological_sort(task_graph))
        except nx.NetworkXUnfeasible:
            # 如果无法进行拓扑排序，使用任意顺序
            topological_order = [t[0] for t in occupation_tasks]
            print(f"警告: 船舶 {vessel} 在泊位 {berth} 的任务优先级图无法进行拓扑排序，使用任意顺序")

        # 初始化任务调度信息
        task_schedule = {}

        # 初始化任务开始和结束时间
        for task_type, workload in occupation_tasks:
            task_schedule[task_type] = {
                'start': 0.0,
                'end': 0.0,
                'duration': workload / instance.vj[task_type]
            }

        # 检查当前占用块中是否存在需要岸桥的任务
        quay_crane_tasks = [task for task in occupation_tasks if task[0] in instance.JQ]

        if quay_crane_tasks:
            # 清理与岸桥相关的决策变量、时间变量、序列变量和辅助变量
            for task_type, workload in quay_crane_tasks:
                # 删除岸桥分配变量 Y
                if (vessel, task_type) in solution.task_to_crane:
                    crane = solution.task_to_crane[(vessel, task_type)]
                    solution.Y[(vessel, task_type, crane)] = 0
                    # 从岸桥序列中删除任务
                    crane_sequence = solution.crane_sequences[crane]
                    for i, crane_task in enumerate(crane_sequence):
                        if crane_task['vessel'] == vessel and crane_task['task'] == task_type:
                            del crane_sequence[i]
                            break
                    # 删除任务到岸桥的映射
                    del solution.task_to_crane[(vessel, task_type)]

        # 调度任务（考虑优先级和岸桥约束）
        for task_type in topological_order:
            # 获取任务持续时间
            task_info = next((t for t_type, t in task_schedule.items() if t_type == task_type), None)
            if not task_info:
                continue

            duration = task_info['duration']

            # 确定最早开始时间
            earliest_start = start_time

            # 考虑优先级约束（前置任务必须完成）
            predecessors = list(task_graph.predecessors(task_type))
            for pred in predecessors:
                if pred in task_schedule:
                    pred_end = task_schedule[pred]['end']
                    if pred_end > earliest_start:
                        earliest_start = pred_end

            # 考虑不在当前占用块内的前置任务
            for (j_prime, j) in priority_edges:
                if j == task_type and j_prime not in occupation_task_types:
                    predecessor_end = None
                    if (vessel, j_prime) in solution.task_end:
                        predecessor_end = solution.task_end[(vessel, j_prime)]
                    else:
                        predecessor_berth = solution.task_to_berth.get((vessel, j_prime))
                        if predecessor_berth is not None:
                            predecessor_end = solution.berth_occupation_end.get((vessel, predecessor_berth))

                    if predecessor_end is not None and predecessor_end > earliest_start:
                        earliest_start = predecessor_end

            # 如果需要岸桥，分配岸桥
            if task_type in instance.JQ:
                # 选择最佳岸桥
                best_crane, best_start = self.select_best_crane(solution, earliest_start, duration)
                '''
                if best_crane is None:
                    # 如果没有找到合适的岸桥，选择负载最轻的岸桥
                    crane_workloads = {}
                    for crane in instance.Q:
                        total_workload = sum(
                            task['end'] - task['start'] for task in solution.crane_sequences.get(crane, []))
                        crane_workloads[crane] = total_workload

                    best_crane = min(crane_workloads, key=crane_workloads.get)
                    best_start = max(earliest_start,
                                     solution.crane_sequences[best_crane][-1]['end'] if solution.crane_sequences[
                                         best_crane] else earliest_start)
                    '''
                # 设置任务开始时间
                task_start = best_start
                task_end = task_start + duration

                # 记录岸桥分配
                solution.Y[(vessel, task_type, best_crane)] = 1
                solution.task_to_crane[(vessel, task_type)] = best_crane

                # 更新岸桥序列
                new_task = {
                    'vessel': vessel,
                    'task': task_type,
                    'start': task_start,
                    'end': task_end
                }
                if best_crane not in solution.crane_sequences:
                    solution.crane_sequences[best_crane] = []
                solution.crane_sequences[best_crane].append(new_task)
                solution.crane_sequences[best_crane].sort(key=lambda x: x['start'])
            else:
                # 不需要岸桥的任务可以立即开始
                task_start = earliest_start
                task_end = task_start + duration

            # 保存任务时间
            task_schedule[task_type]['start'] = task_start
            task_schedule[task_type]['end'] = task_end

            # 更新解中的任务时间
            solution.task_start[(vessel, task_type)] = task_start
            solution.task_end[(vessel, task_type)] = task_end

        # 计算占用结束时间（所有任务结束时间的最大值）
        occupation_end = max(task['end'] for task in task_schedule.values()) if task_schedule else start_time

        return occupation_end

    def select_best_crane(self, solution, earliest_start, duration):
        """选择最适合的岸桥（最早可用且能容纳任务）"""
        instance = self.instance
        best_crane = None
        best_start = float('inf')

        for crane in instance.Q:
            # 获取岸桥的可用时间段（考虑duration）
            available_start, available_end = self.get_crane_availability(solution, crane, earliest_start, duration)

            # 如果找到了可用时间段，并且开始时间更早，则更新
            if available_start != float('inf') and available_start < best_start:
                best_start = available_start
                best_crane = crane

        return best_crane, best_start

    def get_crane_availability(self, solution, crane, earliest_start, duration):
        """获取指定岸桥在指定时间后的第一个可用时间段，该时间段必须至少持续duration时长"""
        # 获取岸桥的当前任务序列
        crane_tasks = solution.crane_sequences.get(crane, [])

        # 如果没有任务，岸桥从一开始就可用，且持续无限时间
        if not crane_tasks:
            return earliest_start, float('inf')

        # 按开始时间排序
        crane_tasks.sort(key=lambda x: x['start'])

        # 检查第一个任务之前的时间段
        if earliest_start + duration <= crane_tasks[0]['start']:
            return earliest_start, crane_tasks[0]['start']

        # 检查任务之间的时间段
        for i in range(len(crane_tasks) - 1):
            current_end = crane_tasks[i]['end']
            next_start = crane_tasks[i + 1]['start']

            # 计算空闲时间段的实际开始时间：取earliest_start和当前结束时间的最大值
            start_candidate = max(earliest_start, current_end)
            if start_candidate + duration <= next_start:
                return start_candidate, next_start

        # 检查最后一个任务之后的时间段
        last_end = crane_tasks[-1]['end']
        start_candidate = max(earliest_start, last_end)
        # 最后一个任务之后的时间段是无限长的，所以只要start_candidate+duration<=inf（总是成立），返回start_candidate和inf
        return start_candidate, float('inf')

    # ================== 破坏算子 ==================
    def random_removal_single(self, solution):
        """随机删除一个任务"""

        # 收集所有任务
        all_tasks = []
        for ship in self.instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                all_tasks.append((vessel, task_type))

        if not all_tasks:
            return solution

        # 随机选择一个任务
        vessel, task_type = random.choice(all_tasks)

        # 输出解的所有详细信息
        # print_solution_details(solution)

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除任务
        self.remove_task(new_solution, vessel, task_type)

        # 输出解的所有详细信息
        # print_solution_details(new_solution)

        return new_solution

    def random_removal_three(self, solution):
        """随机删除三个任务"""
        # 输出解的所有详细信息
        # print_solution_details(solution)

        # 收集所有任务
        all_tasks = []
        for ship in self.instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type = task[0]
                all_tasks.append((vessel, task_type))

        if len(all_tasks) < 3:
            return solution

        # 随机选择三个不同任务
        selected_tasks = random.sample(all_tasks, 3)

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除任务
        for vessel, task_type in selected_tasks:
            self.remove_task(new_solution, vessel, task_type)

        # 输出解的所有详细信息
        # print_solution_details(new_solution)

        return new_solution

    def random_removal_vessel(self, solution):
        """随机删除一艘船舶的所有任务"""
        # 随机选择一艘船舶
        vessel = random.choice([s['vessel'] for s in self.instance.ships])

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除该船舶的所有任务
        tasks = self.instance.get_vessel_tasks(vessel)
        for task in tasks:
            task_type = task[0]
            self.remove_task(new_solution, vessel, task_type)

        return new_solution

    def delay_based_removal_single(self, solution):
        """删除延迟最大的一个任务"""
        # 计算每个任务的延迟贡献
        task_delays = []
        for ship in self.instance.ships:
            vessel = ship['vessel']
            tvd = ship['tvd']  # 获取船舶的预计离泊时间
            for task in ship['tasks']:
                task_type = task[0]
                # 计算任务延迟贡献
                task_end = solution.task_end.get((vessel, task_type), 0)
                delay = max(0, task_end - tvd)
                task_delays.append(((vessel, task_type), delay))

        if not task_delays:
            return solution

        # 选择延迟最大的任务
        max_task = max(task_delays, key=lambda x: x[1])[0]

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除任务
        self.remove_task(new_solution, max_task[0], max_task[1])

        return new_solution

    def delay_based_removal_three(self, solution):
        """删除延迟最大的三个任务"""
        # 计算每个任务的延迟贡献
        task_delays = []
        for ship in self.instance.ships:
            vessel = ship['vessel']
            tvd = ship['tvd']  # 获取船舶的预计离泊时间
            for task in ship['tasks']:
                task_type = task[0]
                # 计算任务延迟贡献
                task_end = solution.task_end.get((vessel, task_type), 0)
                delay = max(0, task_end - tvd)
                task_delays.append(((vessel, task_type), delay))

        if len(task_delays) < 3:
            return solution

        # 选择延迟最大的三个任务
        sorted_tasks = sorted(task_delays, key=lambda x: x[1], reverse=True)
        selected_tasks = [item[0] for item in sorted_tasks[:3]]

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除任务
        for vessel, task_type in selected_tasks:
            self.remove_task(new_solution, vessel, task_type)

        return new_solution

    def delay_based_removal_vessel(self, solution):
        """删除延迟最大的一艘船舶的所有任务"""
        # 找到延迟最大的船舶
        max_vessel = None
        max_delay = -1
        for ship in self.instance.ships:
            vessel = ship['vessel']
            delay = solution.delay.get(vessel, 0)
            if delay > max_delay:
                max_delay = delay
                max_vessel = vessel

        if max_vessel is None:
            return solution

        # 创建解的深拷贝
        new_solution = copy.deepcopy(solution)

        # 移除该船舶的所有任务
        tasks = self.instance.get_vessel_tasks(max_vessel)
        for task in tasks:
            task_type = task[0]
            self.remove_task(new_solution, max_vessel, task_type)

        return new_solution

    def remove_task(self, solution, vessel, task_type):
        """从解中移除指定任务"""
        # 获取任务所在的泊位
        if (vessel, task_type) in solution.task_to_berth:
            berth = solution.task_to_berth[(vessel, task_type)]
            original_start_time = solution.berth_occupation_start.get((vessel, berth), 0)

            # 调用场景1重新调度函数
            self.reschedule_after_removal(solution, vessel, task_type, berth, original_start_time)

            # 重新计算总成本
            # solution.cost = solution.calculate_cost()

    def reschedule_after_removal(self, solution, vessel, task_type, berth, original_start_time):
        """
        场景1：删除任务后的重新调度

        参数:
        solution -- 当前解
        vessel -- 船舶ID
        task_type -- 被删除的任务类型
        berth -- 泊位ID
        original_start_time -- 原占用块的开始时间（作为阈值时间）

        返回:
        重新调度后的解
        """
        instance = self.instance

        # 步骤1: 删除、修改或初始化具有任务i索引的相关的决策变量、时间变量、序列变量和辅助数据结构
        # 删除泊位分配
        solution.X[(vessel, task_type, berth)] = 0
        if (vessel, task_type) in solution.task_to_berth:
            del solution.task_to_berth[(vessel, task_type)]

        # 如果删除的任务需要岸桥，处理岸桥相关变量
        if task_type in instance.JQ and (vessel, task_type) in solution.task_to_crane:
            crane = solution.task_to_crane[(vessel, task_type)]
            # 删除岸桥分配
            solution.Y[(vessel, task_type, crane)] = 0
            # 从岸桥序列中删除任务
            crane_sequence = solution.crane_sequences[crane]
            for i, crane_task in enumerate(crane_sequence):
                if crane_task['vessel'] == vessel and crane_task['task'] == task_type:
                    del crane_sequence[i]
                    break
            del solution.task_to_crane[(vessel, task_type)]

        # 删除时间变量
        if (vessel, task_type) in solution.task_start:
            del solution.task_start[(vessel, task_type)]
        if (vessel, task_type) in solution.task_end:
            del solution.task_end[(vessel, task_type)]

        # 步骤2: 查找船舶在泊位上的占用块
        occupation = None
        occupation_index = -1
        for i, occ in enumerate(solution.berth_sequences[berth]):
            if occ['vessel'] == vessel:
                occupation = occ
                occupation_index = i
                break

        if not occupation:
            # 如果没有找到占用块，说明船舶在该泊位没有其他任务
            # 更新相关决策变量
            solution.Z[(vessel, berth)] = 0
            # 从船舶移泊序列中删除该泊位
            if vessel in solution.vessel_movements:
                if berth in solution.vessel_movements[vessel]:
                    solution.vessel_movements[vessel].remove(berth)
            # 跳转到步骤3
            return self.global_reschedule_after_removal(solution, berth, original_start_time)

        # 从占用块中删除任务
        occupation['tasks'] = [t for t in occupation['tasks'] if t[0] != task_type]

        # 检查占用块中是否还有其他任务
        if not occupation['tasks']:
            # 如果没有其他任务，删除整个占用块
            del solution.berth_sequences[berth][occupation_index]
            solution.Z[(vessel, berth)] = 0
            # 从船舶移泊序列中删除该泊位
            if vessel in solution.vessel_movements:
                if berth in solution.vessel_movements[vessel]:
                    solution.vessel_movements[vessel].remove(berth)
            # 跳转到步骤3
            return self.global_reschedule_after_removal(solution, berth, original_start_time)

        # 如果有其他任务，执行a)~c)
        # a) 岸桥决策变量修改：循环Kbv中的任务，判断Kbv中的任务是否有需要岸桥的任务j
        quay_crane_tasks = [t for t in occupation['tasks'] if t[0] in instance.JQ]

        if quay_crane_tasks:
            # 删除所有需要岸桥的任务的岸桥分配
            for task in quay_crane_tasks:
                task_t = task[0]
                if (vessel, task_t) in solution.task_to_crane:
                    crane = solution.task_to_crane[(vessel, task_t)]
                    solution.Y[(vessel, task_t, crane)] = 0
                    # 从岸桥序列中删除任务
                    crane_sequence = solution.crane_sequences[crane]
                    for i, crane_task in enumerate(crane_sequence):
                        if crane_task['vessel'] == vessel and crane_task['task'] == task_t:
                            del crane_sequence[i]
                            break
                    del solution.task_to_crane[(vessel, task_t)]

        # b) Kbv中的任务调度
        tasks = occupation['tasks']
        start_time = occupation['start']

        # 使用新的岸桥调度逻辑
        end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time)

        # 更新占用结束时间
        old_end = occupation['end']
        occupation['end'] = end_time
        solution.berth_occupation_end[(vessel, berth)] = end_time

        # c) 判断调度是否终止
        if abs(end_time - old_end) < 1e-6:
            # 结束时间没有变化，不需要进一步调度
            return solution
        else:
            # 结束时间有变化，跳转到步骤3
            return self.global_reschedule_after_removal(solution, berth, original_start_time)

    def global_reschedule_after_removal(self, solution, berth, threshold_time):
        """
        场景1的步骤3：全局重新调度
        """
        instance = self.instance

        # (1) 重新设置各个资源可用时间
        # 泊位资源重新设置
        berth_availability = {}
        for b in instance.berths:
            # 找到结束时间小于等于阈值时间的最大结束时间
            max_end_time = 0
            for occ in solution.berth_sequences[b]:
                if occ['end'] <= threshold_time and occ['end'] > max_end_time:
                    max_end_time = occ['end']
            berth_availability[b] = max_end_time

        # 岸桥资源重新设置
        crane_availability = {}
        for crane in instance.Q:
            # 找到结束时间小于等于阈值时间的最大结束时间
            max_end_time = 0
            for task in solution.crane_sequences[crane]:
                if task['end'] <= threshold_time and task['end'] > max_end_time:
                    max_end_time = task['end']
            crane_availability[crane] = max_end_time

        # (2) 调度计划初始化
        # 对于结束时间大于阈值时间的占用，初始化时间相关的决策变量和岸桥相关的决策变量
        for b in instance.berths:
            for occ in solution.berth_sequences[b]:
                if occ['end'] > threshold_time:
                    vessel = occ['vessel']
                    # 初始化时间变量
                    for task in occ['tasks']:
                        task_t = task[0]
                        if (vessel, task_t) in solution.task_start:
                            del solution.task_start[(vessel, task_t)]
                        if (vessel, task_t) in solution.task_end:
                            del solution.task_end[(vessel, task_t)]

                    # 初始化岸桥变量
                    for task in occ['tasks']:
                        task_t = task[0]
                        if task_t in instance.JQ and (vessel, task_t) in solution.task_to_crane:
                            crane = solution.task_to_crane[(vessel, task_t)]
                            solution.Y[(vessel, task_t, crane)] = 0
                            # 从岸桥序列中删除任务
                            crane_sequence = solution.crane_sequences[crane]
                            for i, crane_task in enumerate(crane_sequence):
                                if crane_task['vessel'] == vessel and crane_task['task'] == task_t:
                                    del crane_sequence[i]
                                    break
                            del solution.task_to_crane[(vessel, task_t)]

        # (3) 重新调度
        # 步骤1：创建重调度集合CDDS
        CDDS = []
        for b in instance.berths:
            for occ in solution.berth_sequences[b]:
                if occ['end'] > threshold_time:
                    CDDS.append(occ)

        '''
        # ============ 添加的代码：打印CDDS信息 ============
        print("\n=== 重调度集合 CDDS ===")
        print(f"阈值时间: {threshold_time}")
        print("需要重调度的占用块:")
        for i, occ in enumerate(CDDS):
            print(f"{i + 1}. 泊位: {berth}, 船舶: {occ['vessel']}, "
                  f"开始时间: {occ['start']:.2f}, 结束时间: {occ['end']:.2f}, "
                  f"任务: {[task[0] for task in occ['tasks']]}")
        print("=" * 50)

        def sort_occupations(occupation1, occupation2, instance):
            """
            排序函数：首先按开始时间升序排序，对于开始时间相同的同一船舶的不同占用，
            根据任务优先级关系排序（有优先关系的排在前面）
            """
            # 第一排序准则：开始时间
            if occupation1['start'] < occupation2['start']:
                return -1
            elif occupation1['start'] > occupation2['start']:
                return 1
            else:
                # 开始时间相同，检查是否是同一船舶的不同占用
                vessel1 = occupation1['vessel']
                vessel2 = occupation2['vessel']

                # 只有同一船舶的不同占用才可能有优先级关系
                if vessel1 == vessel2:
                    # 获取该船舶的优先级约束
                    priority_edges = instance.get_priority_constraints(vessel1)

                    # 检查两个占用之间的任务优先级关系
                    for task1 in occupation1['tasks']:
                        task_type1 = task1[0]
                        for task2 in occupation2['tasks']:
                            task_type2 = task2[0]
                            # 如果存在优先级关系 (task_type1, task_type2)，则occupation1排在前面
                            if (task_type1, task_type2) in priority_edges:
                                return -1
                            # 如果存在优先级关系 (task_type2, task_type1)，则occupation2排在前面
                            elif (task_type2, task_type1) in priority_edges:
                                return 1

                    # 如果没有找到明确的优先级关系，保持原顺序
                    return 0
                else:
                    # 不同船舶的占用，开始时间相同，保持原顺序
                    return 0

        # 使用自定义排序函数进行排序，传递 instance 参数
        #SCDDS = sorted(CDDS, key=functools.cmp_to_key(lambda x, y: sort_occupations(x, y, instance)))
        # 根据开始时间升序排序
        '''

        # CDDS = sorted(CDDS, key=lambda x: x['start'])

        def get_berth_of_occ(occupation):
            for berth_id in instance.berths:
                if occupation in solution.berth_sequences[berth_id]:
                    return berth_id
            return None

        SCDDS = sorted(
            CDDS,
            key=lambda occ: (
                solution.vessel_movements.get(occ['vessel'], []).index(get_berth_of_occ(occ))
                if get_berth_of_occ(occ) in solution.vessel_movements.get(occ['vessel'], []) else 0,
                occ['start']
            )
        )

        '''
        # ============ 添加的代码：打印CDDS信息 ============
        print("\n=== 重调度集合 CDDS 排序===")
        print(f"阈值时间: {threshold_time}")
        print("需要重调度的占用块:")
        for i, occ in enumerate(CDDS):
            print(f"{i + 1}. 泊位: {berth}, 船舶: {occ['vessel']}, "
                  f"开始时间: {occ['start']:.2f}, 结束时间: {occ['end']:.2f}, "
                  f"任务: {[task[0] for task in occ['tasks']]}")
        print("=" * 50)
        # ============ 结束添加的代码 ============
        '''

        # 使用元组排序
        '''
        SCDDS = sorted(CDDS, key=lambda x: (
            x['start'],  # 第一排序准则：开始时间
            -get_priority_score(x, instance)  # 第二排序准则：优先级分数（降序）
        ))
        '''

        # 步骤2：遍历SCDDS中的元素
        for occ in SCDDS:
            vessel = occ['vessel']
            b = None
            # 查找泊位b
            for berth in instance.berths:
                if occ in solution.berth_sequences[berth]:
                    b = berth
                    break

            if b is None:
                continue

            tasks = occ['tasks']

            # 获取上一个船舶占位的结束时间
            prev_occupation_end = berth_availability[b]

            # 获取船舶上一个泊位占位的结束时间+移泊时间
            berth_sequence = solution.vessel_movements[vessel]
            berth_index = berth_sequence.index(b)
            if berth_index == 0:
                prev_berth_end = self.get_previous_berth_end_time(solution, vessel, b)
            else:
                prev_berth_end = self.get_previous_berth_end_time(solution, vessel, b) + instance.tvc

            '''
            # 考虑跨泊位的优先级约束
            priority_constraint_start = 0
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                # 检查是否有任务在当前占用中，而其前置任务在其他泊位
                j_in_current = any(t[0] == j for t in tasks)
                j_prime_in_other = (vessel, j_prime) in solution.task_end and not any(t[0] == j_prime for t in tasks)

                if j_in_current and j_prime_in_other:
                    # 前置任务在其他泊位，需要等待其完成
                    j_prime_end = solution.task_end[(vessel, j_prime)]
                    if j_prime_end > priority_constraint_start:
                        priority_constraint_start = j_prime_end

            # 计算开始时间（考虑所有约束）
            start_time = max(prev_occupation_end, prev_berth_end, priority_constraint_start)
            '''

            # 获取船舶的预计到达时间
            ship_data = next((s for s in self.instance.ships if s['vessel'] == vessel), None)
            if ship_data:
                tva = ship_data['tva']
            else:
                tva = 0  # 默认值，理论上不应该发生

            # 考虑跨泊位的优先级约束
            priority_constraint_start = 0.0
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                if any(t[0] == j for t in tasks):
                    # 若前置任务在当前占用中，由schedule_berth_occupation内部处理
                    if any(t[0] == j_prime for t in tasks):
                        continue

                    predecessor_end = None
                    berth_prime = solution.task_to_berth.get((vessel, j_prime))
                    if berth_prime is not None:
                        predecessor_end = solution.berth_occupation_end.get((vessel, berth_prime))

                    if predecessor_end is None and (vessel, j_prime) in solution.task_end:
                        predecessor_end = solution.task_end[(vessel, j_prime)]

                    if predecessor_end is not None and predecessor_end > priority_constraint_start:
                        priority_constraint_start = predecessor_end

            # 计算开始时间
            start_time = max(prev_occupation_end, prev_berth_end, priority_constraint_start, tva)

            # start_time = max(prev_occupation_end, prev_berth_end)

            # 调度占用
            end_time = self.schedule_berth_occupation(vessel, b, tasks, solution, start_time)

            # 更新占用信息
            occ['start'] = start_time
            occ['end'] = end_time
            solution.berth_occupation_start[(vessel, b)] = start_time
            solution.berth_occupation_end[(vessel, b)] = end_time

            # 根据最新时间更新排序与移泊顺序
            solution.berth_sequences[b].sort(key=lambda x: x['start'])
            self.update_vessel_movement_order(solution, vessel)

            # 更新资源可用时间
            berth_availability[b] = end_time

            # 更新岸桥可用时间（这里简化处理，实际应该从调度过程中获取）
            # 这里需要根据实际调度情况更新crane_availability

        return solution
    '''
    def update_vessel_movement_order(self, solution, vessel):
        """按照最新的开始时间对船舶的移泊顺序重新排序"""
        if vessel not in solution.vessel_movements:
            return

        default_start = next(
            (ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel),
            0.0
        )

        def movement_key(berth_id):
            start = solution.berth_occupation_start.get((vessel, berth_id), default_start)
            return (
                start,
                self.get_priority_score(solution, vessel, berth_id)
            )

        solution.vessel_movements[vessel].sort(key=movement_key)
    '''
    def update_vessel_movement_order(self, solution, vessel):
        """按照最新的开始时间和优先级约束对船舶的移泊顺序重新排序"""
        if vessel not in solution.vessel_movements:
            return

        berth_sequence = solution.vessel_movements[vessel]
        if not berth_sequence:
            return

        default_start = next(
            (ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel),
            0.0
        )

        def movement_key(berth_id):
            start = solution.berth_occupation_start.get((vessel, berth_id), default_start)
            return (start, berth_id)

        graph = nx.DiGraph()
        graph.add_nodes_from(berth_sequence)

        for predecessor, successor in self.instance.get_priority_constraints(vessel):
            from_berth = solution.task_to_berth.get((vessel, predecessor))
            to_berth = solution.task_to_berth.get((vessel, successor))

            if from_berth is None or to_berth is None:
                continue

            if from_berth not in graph or to_berth not in graph:
                continue

            if from_berth == to_berth:
                continue

            graph.add_edge(from_berth, to_berth)

        try:
            ordered_berths = list(
                nx.algorithms.dag.lexicographical_topological_sort(graph, key=movement_key)
            )
        except nx.NetworkXUnfeasible:
            ordered_berths = sorted(berth_sequence, key=movement_key)

        solution.vessel_movements[vessel] = ordered_berths

    def get_previous_berth_end_time(self, solution, vessel, berth):
        """
        获取船舶在指定泊位前一个泊位的结束时间

        参数:
        solution -- 当前解
        vessel -- 船舶ID
        berth -- 泊位ID

        返回:
        前一个泊位的结束时间，如果没有前一个泊位，返回船舶到达时间
        """
        if vessel not in solution.vessel_movements:
            return next(ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel)

        berth_sequence = solution.vessel_movements[vessel]
        if berth not in berth_sequence:
            return next(ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel)

        berth_index = berth_sequence.index(berth)
        if berth_index == 0:
            return next(ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel)

        prev_berth = berth_sequence[berth_index - 1]
        return solution.berth_occupation_end.get((vessel, prev_berth), next(
            ship['tva'] for ship in self.instance.ships if ship['vessel'] == vessel))

    # ================== 修复算子 ==================
    def random_insertion(self, solution):
        """随机插入被移除的任务（支持并行任务）"""

        # 输出解的所有详细信息
        # print("1rr---------------------")
        print_solution_details(solution)
        # print("1r---------------------")

        # 输出解的所有详细信息
        # print_solution_details(solution)

        # 收集所有被移除的任务（即当前解中缺失的任务）
        missing_tasks = self.get_missing_tasks(solution)

        # 随机打乱任务顺序
        # random.shuffle(missing_tasks)
        # 按优先级约束排序任务
        sorted_missing = self.sort_tasks_by_priority(missing_tasks)

        # 逐个插入任务
        for vessel, task_type, workload in sorted_missing:
            # 获取时间阈值T
            time_threshold = self.get_time_threshold(solution, vessel, task_type)
            time_threshold_S = self.get_time_threshold_S(solution, vessel, task_type)

            # 获取可能的泊位
            possible_berths = self.instance.Bj[task_type]
            if not possible_berths:
                print(f"错误: 任务 {task_type} 没有可用的泊位")
                continue

            # 随机打乱泊位顺序
            random.shuffle(possible_berths)

            success = False

            # 尝试每个可能的泊位
            for berth in possible_berths:
                # 步骤1: 检查互斥约束
                if not self.check_mutual_exclusion(solution, vessel, task_type, berth):
                    continue

                # 步骤2: 判断泊位b是否存在船舶v的占用
                existing_occupation = None
                for occ in solution.berth_sequences[berth]:
                    if occ['vessel'] == vessel:
                        existing_occupation = occ
                        break
                '''
                if existing_occupation:
                    # 获取当前任务j的优先任务
                    priority_edges = self.instance.get_priority_constraints(vessel)
                    predecessor_tasks = set()
                    for (j_prime, j_val) in priority_edges:
                        if j_val == task_type:
                            predecessor_tasks.add(j_prime)

                    # 检查当前泊位b上船舶v的占用是否包含优先任务
                    has_predecessor = False
                    if existing_occupation['tasks']:
                        for task in existing_occupation['tasks']:
                            if task[0] in predecessor_tasks:
                                has_predecessor = True
                                break

                    # 获取当前任务j的后续任务
                    priority_edges = self.instance.get_priority_constraints(vessel)
                    successor_tasks = set()
                    for (j_prime, j_val) in priority_edges:
                        if j_prime == task_type:  # 当前任务是这些任务的前置任务
                            successor_tasks.add(j_val)

                    # 检查当前泊位b上船舶v的占用是否包含后续任务
                    has_successor = False
                    if existing_occupation['tasks']:
                        for task in existing_occupation['tasks']:
                            if task[0] in successor_tasks:
                                has_successor = True
                                break
                '''
                # print("rondom_insertion---------------------")
                '''
                if existing_occupation:
                    # 存在占用，检查开始时间是否大于等于时间阈值T
                    if (occ['start'] >= time_threshold and occ[
                        'end'] < time_threshold_S) or has_predecessor or has_successor:
                        # 创建临时解副本
                        temp_solution = copy.deepcopy(solution)
                        # 尝试插入任务
                        if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                            # 如果插入成功，更新解并跳出循环
                            solution.__dict__.update(temp_solution.__dict__)
                            success = True
                            break
                    else:
                        continue
                '''

                if existing_occupation:
                    temp_solution = copy.deepcopy(solution)
                    if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                        solution.__dict__.update(temp_solution.__dict__)
                        success = True
                        break
                else:
                    # 不存在占用，创建子集合SBZY
                    sub_occupations = []
                    for occ in solution.berth_sequences[berth]:
                        if occ['start'] >= time_threshold and occ['end'] < time_threshold_S:
                            sub_occupations.append(occ)

                    # if not sub_occupations:
                    # continue

                    # 生成所有可能的位置索引 (0: 第一个元素之前, 1: 第一个和第二个之间, ..., n: 最后一个元素之后)
                    n = len(sub_occupations)
                    all_positions = list(range(n + 1))
                    random.shuffle(all_positions)

                    success_in_sub = False
                    for pos_index in all_positions:
                        # 创建临时解副本
                        temp_solution = copy.deepcopy(solution)

                        # 创建新的占用
                        if self.create_new_occupation(temp_solution, vessel, berth, pos_index, sub_occupations,
                                                      time_threshold, time_threshold_S, task_type, workload):
                            # 尝试插入任务
                            if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                                solution.__dict__.update(temp_solution.__dict__)
                                success_in_sub = True
                                break

                    if success_in_sub:
                        success = True
                        break

            # 输出解的所有详细信息
            # print("3rr---------------------")
            # print_solution_details(solution)
            # print("3r---------------------")

            if not success:
                print(f"警告: 无法插入任务 ({vessel}, {task_type})，尝试了所有可能的泊位")

            # 输出解的所有详细信息
            # print("4rr---------------------")
            # print_solution_details(solution)
            # print("4r---------------------")

            # 插入所有任务后，更新所有船舶的结束时间和延迟
            self.update_vessel_end_times_and_delays(solution)

            # 使用calculate_cost函数计算总成本
            solution.cost = solution.calculate_cost()

        # 输出解的所有详细信息
        # print_solution_details(solution)

        return solution

    def update_vessel_end_times_and_delays(self, solution):
        """更新所有船舶的结束时间和延迟"""
        for ship in self.instance.ships:
            vessel = ship['vessel']
            # 获取船舶在所有泊位上的最晚结束时间
            vessel_end = 0.0
            for b in self.instance.berths:
                if (vessel, b) in solution.berth_occupation_end:
                    end_time = solution.berth_occupation_end[(vessel, b)]
                    if end_time > vessel_end:
                        vessel_end = end_time

            # 更新船舶结束时间
            solution.vessel_end[vessel] = vessel_end

            # 计算船舶延迟
            tvd = ship['tvd']
            delay = max(0, vessel_end - tvd)
            solution.delay[vessel] = delay

    def get_time_threshold(self, solution, vessel, task_type):
        """获取任务的时间阈值T（前置任务的最晚结束时间）"""
        instance = self.instance
        # 初始阈值设为船舶的预计到达时间

        ship_data = next((s for s in instance.ships if s['vessel'] == vessel), None)
        if ship_data:
            threshold = ship_data['tva']  # 船舶预计到达时间
        else:
            threshold = 0.0  # 默认值，理论上不应该发生

        # threshold = 0.0

        # 获取该船舶的优先级约束
        priority_edges = instance.get_priority_constraints(vessel)

        # 找出所有需要在当前任务之前完成的任务
        predecessor_tasks = []
        for (j_prime, j) in priority_edges:
            if j == task_type:
                predecessor_tasks.append(j_prime)

        # 计算最晚结束时间
        for pred_task in predecessor_tasks:
            if (vessel, pred_task) in solution.task_end:
                end_time = solution.task_end[(vessel, pred_task)]
                if end_time > threshold:
                    threshold = end_time
        '''
        # 考虑船舶在泊位上的占用开始时间
        for berth in instance.berths:
            if (vessel, berth) in solution.berth_occupation_start:
                start_time = solution.berth_occupation_start[(vessel, berth)]
                if start_time > threshold:
                    threshold = start_time
        '''
        return threshold

    def get_time_threshold_S(self, solution, vessel, task_type):
        """获取任务的时间阈值S（后续任务的最早开始时间）"""
        instance = self.instance
        threshold = float('inf')  # 初始化为无穷大，因为我们找最小值

        # 获取该船舶的优先级约束
        priority_edges = instance.get_priority_constraints(vessel)

        # 找出所有需要在当前任务之后开始的任务
        successor_tasks = []
        for (j_prime, j) in priority_edges:
            if j_prime == task_type:  # 当前任务是这些任务的前置任务
                successor_tasks.append(j)

        # 计算最早开始时间
        for succ_task in successor_tasks:
            if (vessel, succ_task) in solution.task_start:
                start_time = solution.task_start[(vessel, succ_task)]
                if start_time < threshold:
                    threshold = start_time

        return threshold

    def check_mutual_exclusion(self, solution, vessel, task_type, berth):
        """检查互斥约束"""
        instance = self.instance

        if task_type in instance.JE:
            # 互斥任务：泊位上不能有其他任务（同一船舶）
            for occ in solution.berth_sequences[berth]:
                if occ['vessel'] == vessel and len(occ['tasks']) > 0:
                    return False
        else:
            # 非互斥任务：泊位上不能有互斥任务（同一船舶）
            for occ in solution.berth_sequences[berth]:
                if occ['vessel'] == vessel:
                    for task in occ['tasks']:
                        if task[0] in instance.JE:
                            return False
        return True

    def create_new_occupation(self, solution, vessel, berth, position_index, sub_occupations, time_threshold,
                              time_threshold_S, task_type, workload):
        """创建新的占用

        Args:
            solution: 当前解
            vessel: 船舶ID
            berth: 泊位ID
            position_index: 位置索引 (0: 第一个元素之前, 1: 第一个和第二个之间, ..., n-1: 最后一个元素之后)
            sub_occupations: 子集合中的占用列表
            time_threshold: 时间阈值
        """
        '''
        n = len(sub_occupations)
        if position_index < 0 or position_index > n:
            return False

        # 确定开始时间
        if n == 0:
            if time_threshold > solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S == float('inf'):
                # 子集合为空，开始时间为0
                start_time = max(solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'], time_threshold)
            elif time_threshold == solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S == float('inf'):
                start_time = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
            elif time_threshold == solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S < float('inf') and time_threshold_S > \
                    solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']:
                start_time = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
            else:
                return False

        elif position_index == 0:
            # if position_index == 0:
            # 第一个元素之前
            # if time_threshold > 0 and time_threshold_S == float('inf'):
            # start_time = max(solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'],time_threshold)  # 假设第一个占用之前的时间为0
            # else:
            # start_time = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
            if time_threshold > solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S == float('inf'):
                # 子集合为空，开始时间为0
                start_time = max(solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'], time_threshold)
            elif time_threshold == solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S == float('inf'):
                start_time = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
            elif time_threshold == solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva'] and time_threshold_S < float('inf') and time_threshold_S > \
                    solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']:
                start_time = solution.instance.ships[self.instance.vessel_to_index[vessel]]['tva']
            else:
                return False

        elif position_index == n:
            # 最后一个元素之后
            start_time = sub_occupations[-1]['end']
        else:
            # 两个元素之间
            start_time = sub_occupations[position_index - 1]['end']
        '''

        n = len(sub_occupations)
        if position_index < 0 or position_index > n:
            return False

        ship_data = next((s for s in self.instance.ships if s['vessel'] == vessel), None)
        arrival_time = ship_data['tva'] if ship_data else 0.0

        earliest_allowed = max(arrival_time, time_threshold)
        latest_allowed = time_threshold_S

        if position_index == 0:
            prev_end = earliest_allowed
        else:
            prev_end = max(earliest_allowed, sub_occupations[position_index - 1]['end'])

        start_time = prev_end

        #if latest_allowed != float('inf') and start_time >= latest_allowed:
            #return False

        # 创建新的占用
        new_occupation = {
            'vessel': vessel,
            'start': start_time,
            'end': start_time,  # 初始结束时间等于开始时间，插入任务后会更新
            'tasks': []
        }

        # 插入到泊位序列中
        solution.berth_sequences[berth].append(new_occupation)
        solution.berth_sequences[berth].sort(key=lambda x: x['start'])

        # 更新决策变量
        solution.Z[(vessel, berth)] = 1
        solution.berth_occupation_start[(vessel, berth)] = start_time
        solution.berth_occupation_end[(vessel, berth)] = start_time

        # 更新船舶移泊序列
        if vessel not in solution.vessel_movements:
            solution.vessel_movements[vessel] = []
        if berth not in solution.vessel_movements[vessel]:
            solution.vessel_movements[vessel].append(berth)

        # new_occupation['tasks'].append((task_type, workload))

        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] == vessel:
                occ['tasks'].append((task_type, workload))
                break
        '''
        # 根据开始时间对移泊序列排序，如果开始时间相同则按任务优先级排序
        solution.vessel_movements[vessel].sort(
            key=lambda b: (
                solution.berth_occupation_start.get((vessel, b), 0),
                self.get_priority_score(solution, vessel, b)
            )
        )
        '''
        self.update_vessel_movement_order(solution, vessel)

        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] == vessel:
                occ['tasks'].remove((task_type, workload))
                break

        # new_occupation['tasks'].append((task_type, workload))

        # 根据开始时间对移泊序列排序
        # solution.vessel_movements[vessel].sort(
        # key=lambda b: solution.berth_occupation_start.get((vessel, b), 0)
        # )

        return True

    def get_priority_score(self, solution, vessel, berth):
        """计算泊位的优先级分数（用于排序）"""
        # 获取该船舶在指定泊位上的所有任务␊
        tasks_in_berth = []
        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] == vessel:
                tasks_in_berth = [task[0] for task in occ['tasks']]
                break

        # 获取该船舶的优先级约束
        priority_edges = self.instance.get_priority_constraints(vessel)

        # 计算优先级分数：具有更高优先级任务的泊位得分更高
        score = 0
        for task in tasks_in_berth:
            # 检查此任务是否是其他任务的前置任务
            for (j_prime, j) in priority_edges:
                if j_prime == task:
                    score += 1  # 是前置任务，加分
            # 检查此任务是否有前置任务
            for (j_prime, j) in priority_edges:
                if j == task:
                    score -= 1  # 有前置任务，减分

        return -score  # 返回负分以便更高优先级的排在前面

    def greedy_insertion(self, solution):
        """贪心插入被移除的任务（最小化延迟，支持并行任务）"""
        # 输出解的所有详细信息
        # print_solution_details(solution)

        # 收集所有被移除的任务（即当前解中缺失的任务）
        missing_tasks = self.get_missing_tasks(solution)

        # 按优先级约束排序任务
        sorted_missing = self.sort_tasks_by_priority(missing_tasks)

        # 逐个插入任务（使用贪心策略）
        for vessel, task_type, workload in sorted_missing:
            # 获取时间阈值T
            time_threshold = self.get_time_threshold(solution, vessel, task_type)
            time_threshold_S = self.get_time_threshold_S(solution, vessel, task_type)

            success = False
            max_attempts = 1  # 最大尝试次数

            for attempt in range(max_attempts):
                # 创建临时解副本
                temp_solution = copy.deepcopy(solution)

                # 尝试插入任务
                if self.insert_task_greedy(temp_solution, vessel, task_type, workload, time_threshold,
                                           time_threshold_S):
                    # 如果插入成功，更新解并跳出循环
                    solution.__dict__.update(temp_solution.__dict__)
                    success = True
                    break

            if not success:
                print(f"1警告: 无法插入任务 ({vessel, task_type})，即使尝试了 {max_attempts} 次")

        # 输出解的所有详细信息
        # print_solution_details(solution)
        # 插入所有任务后，更新所有船舶的结束时间和延迟
        self.update_vessel_end_times_and_delays(solution)

        # 使用calculate_cost函数计算总成本
        solution.cost = solution.calculate_cost()

        return solution

    def regret_insertion(self, solution):
        """后悔值插入被移除的任务（支持并行任务）"""
        # 收集所有被移除的任务（即当前解中缺失的任务）

        missing_tasks = self.get_missing_tasks(solution)

        # 按优先级约束排序任务
        sorted_missing = self.sort_tasks_by_priority(missing_tasks)

        # 计算每个任务的后悔值
        regrets = []
        for vessel, task_type, workload in sorted_missing:
            # 获取时间阈值T
            time_threshold = self.get_time_threshold(solution, vessel, task_type)
            time_threshold_S = self.get_time_threshold_S(solution, vessel, task_type)

            # 评估所有可能的插入位置
            costs = []
            possible_berths = self.instance.Bj[task_type]

            for berth in possible_berths:
                # 步骤1: 检查互斥约束
                if not self.check_mutual_exclusion(solution, vessel, task_type, berth):
                    costs.append(float('inf'))
                    continue

                # 步骤2: 判断泊位b是否存在船舶v的占用
                existing_occupation = None
                for occ in solution.berth_sequences[berth]:
                    if occ['vessel'] == vessel:
                        existing_occupation = occ
                        break
                '''
                if existing_occupation:
                    # 获取当前任务j的优先任务
                    priority_edges = self.instance.get_priority_constraints(vessel)
                    predecessor_tasks = set()
                    for (j_prime, j_val) in priority_edges:
                        if j_val == task_type:
                            predecessor_tasks.add(j_prime)

                    # 检查当前泊位b上船舶v的占用是否包含优先任务
                    has_predecessor = False
                    if existing_occupation['tasks']:
                        for task in existing_occupation['tasks']:
                            if task[0] in predecessor_tasks:
                                has_predecessor = True
                                break

                    # 获取当前任务j的后续任务
                    priority_edges = self.instance.get_priority_constraints(vessel)
                    successor_tasks = set()
                    for (j_prime, j_val) in priority_edges:
                        if j_prime == task_type:  # 当前任务是这些任务的前置任务
                            successor_tasks.add(j_val)

                    # 检查当前泊位b上船舶v的占用是否包含后续任务
                    has_successor = False
                    if existing_occupation['tasks']:
                        for task in existing_occupation['tasks']:
                            if task[0] in successor_tasks:
                                has_successor = True
                                break
                '''
                # print("regret_insertion---------------------")

                '''
                if existing_occupation:
                    # 存在占用，检查开始时间是否大于等于时间阈值T
                    if (occ['start'] >= time_threshold and occ[
                        'end'] < time_threshold_S) or has_predecessor or has_successor:
                        # 创建临时解
                        temp_solution = copy.deepcopy(solution)
                        # 尝试插入
                        if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                            cost = temp_solution.cost
                            costs.append(cost)
                        else:
                            costs.append(float('inf'))
                    else:
                        costs.append(float('inf'))
                '''
                if existing_occupation:
                    temp_solution = copy.deepcopy(solution)
                    if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                        cost = temp_solution.cost
                        costs.append(cost)
                    else:
                        costs.append(float('inf'))
                else:
                    # 不存在占用，创建子集合SBZY
                    sub_occupations = []
                    for occ in solution.berth_sequences[berth]:
                        if occ['start'] >= time_threshold and occ['end'] < time_threshold_S:
                            sub_occupations.append(occ)

                    # if not sub_occupations:
                    # costs.append(float('inf'))
                    # continue

                    # 遍历所有可能的位置索引 (0: 第一个元素之前, 1: 第一个和第二个之间, ..., n: 最后一个元素之后)
                    n = len(sub_occupations)
                    min_cost_for_berth = float('inf')
                    for pos_index in range(n + 1):
                        # 创建临时解
                        temp_solution = copy.deepcopy(solution)

                        # 创建新的占用
                        if self.create_new_occupation(temp_solution, vessel, berth, pos_index, sub_occupations,
                                                      time_threshold, time_threshold_S, task_type, workload):
                            # 尝试插入
                            if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                                cost = temp_solution.cost
                                if cost < min_cost_for_berth:
                                    min_cost_for_berth = cost

                    costs.append(min_cost_for_berth if min_cost_for_berth != float('inf') else float('inf'))

            # 如果没有可行位置，跳过
            if not costs or min(costs) == float('inf'):
                regrets.append(((vessel, task_type, workload), 0))
                continue

            # 计算后悔值（次优成本 - 最优成本）
            sorted_costs = sorted(costs)
            regret = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else 0
            regrets.append(((vessel, task_type, workload), regret))

        # 按后悔值降序排序
        sorted_regrets = sorted(regrets, key=lambda x: x[1], reverse=True)
        sorted_missing = [item[0] for item in sorted_regrets]

        # 逐个插入任务（使用贪心策略）
        for vessel, task_type, workload in sorted_missing:
            success = False
            max_attempts = 1  # 最大尝试次数
            time_threshold = self.get_time_threshold(solution, vessel, task_type)
            time_threshold_S = self.get_time_threshold_S(solution, vessel, task_type)
            for attempt in range(max_attempts):
                # 创建临时解副本
                temp_solution = copy.deepcopy(solution)

                # 尝试插入任务
                if self.insert_task_greedy(temp_solution, vessel, task_type, workload, time_threshold,
                                           time_threshold_S):
                    # 如果插入成功，更新解并跳出循环
                    solution.__dict__.update(temp_solution.__dict__)
                    success = True
                    break

            if not success:
                print(f"2警告: 无法插入任务 ({vessel, task_type})，即使尝试了 {max_attempts} 次")

        # 插入所有任务后，更新所有船舶的结束时间和延迟
        self.update_vessel_end_times_and_delays(solution)

        # 使用calculate_cost函数计算总成本
        solution.cost = solution.calculate_cost()

        return solution

    def get_missing_tasks(self, solution):
        """获取当前解中缺失的任务"""
        missing_tasks = []

        for ship in self.instance.ships:
            vessel = ship['vessel']
            for task in ship['tasks']:
                task_type, workload = task
                # 检查任务是否在解中
                if (vessel, task_type) not in solution.task_to_berth:
                    missing_tasks.append((vessel, task_type, workload))

        return missing_tasks

    def sort_tasks_by_priority(self, tasks):
        """按优先级约束排序任务（确保优先任务先插入）"""
        # 按船舶分组
        vessel_tasks = defaultdict(list)
        for vessel, task_type, workload in tasks:
            vessel_tasks[vessel].append((task_type, workload))

        sorted_tasks = []
        for vessel, tasks_list in vessel_tasks.items():
            # 获取该船舶的优先级约束
            priority_edges = self.instance.get_priority_constraints(vessel)

            # 创建依赖图
            G = nx.DiGraph()
            for task_type, _ in tasks_list:
                G.add_node(task_type)
            for (j_prime, j) in priority_edges:
                if G.has_node(j_prime) and G.has_node(j):
                    G.add_edge(j_prime, j)

            # 拓扑排序
            try:
                ordered = list(nx.topological_sort(G))
                # 按拓扑顺序添加任务
                for task_type in ordered:
                    workload = next(w for t, w in tasks_list if t == task_type)
                    sorted_tasks.append((vessel, task_type, workload))
                # 添加不在图中的任务
                for task_type, workload in tasks_list:
                    if task_type not in ordered:
                        sorted_tasks.append((vessel, task_type, workload))
            except nx.NetworkXUnfeasible:
                # 有环，随机排序
                for task_type, workload in tasks_list:
                    sorted_tasks.append((vessel, task_type, workload))

        return sorted_tasks

    def insert_task_greedy(self, solution, vessel, task_type, workload, time_threshold, time_threshold_S):
        """贪心插入任务（选择最小化延迟的泊位，支持并行任务）"""

        possible_berths = self.instance.Bj[task_type]
        if not possible_berths:
            return False

        best_berth = None
        best_cost = float('inf')
        best_solution = None

        # 评估每个可能的泊位
        for berth in possible_berths:
            # 步骤1: 检查互斥约束
            if not self.check_mutual_exclusion(solution, vessel, task_type, berth):
                continue

            # 步骤2: 判断泊位b是否存在船舶v的占用
            existing_occupation = None
            for occ in solution.berth_sequences[berth]:
                if occ['vessel'] == vessel:
                    existing_occupation = occ
                    break

            '''
            if existing_occupation:
                # 获取当前任务j的优先任务
                priority_edges = self.instance.get_priority_constraints(vessel)
                predecessor_tasks = set()
                for (j_prime, j_val) in priority_edges:
                    if j_val == task_type:
                        predecessor_tasks.add(j_prime)

                # 检查当前泊位b上船舶v的占用是否包含优先任务
                has_predecessor = False
                if existing_occupation['tasks']:
                    for task in existing_occupation['tasks']:
                        if task[0] in predecessor_tasks:
                            has_predecessor = True
                            break

                # 获取当前任务j的后续任务
                priority_edges = self.instance.get_priority_constraints(vessel)
                successor_tasks = set()
                for (j_prime, j_val) in priority_edges:
                    if j_prime == task_type:  # 当前任务是这些任务的前置任务
                        successor_tasks.add(j_val)

                # 检查当前泊位b上船舶v的占用是否包含后续任务
                has_successor = False
                if existing_occupation['tasks']:
                    for task in existing_occupation['tasks']:
                        if task[0] in successor_tasks:
                            has_successor = True
                            break

            '''

            # task_duration = workload / self.instance.vj[task_type]

            # print("greedy_insertion---------------------")
            '''
            if existing_occupation:
                # 存在占用，检查开始时间是否大于等于时间阈值T
                # if (occ['start'] >= time_threshold and occ['end'] <= time_threshold_S and occ[
                # 'start'] + task_duration <= time_threshold_S) or has_predecessor or has_successor:
                if (occ['start'] >= time_threshold and occ[
                    'end'] < time_threshold_S) or has_predecessor or has_successor:
                    # 创建临时解
                    temp_solution = copy.deepcopy(solution)
                    # 尝试插入
                    if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                        if temp_solution.cost < best_cost:
                            best_cost = temp_solution.cost
                            best_berth = berth
                            best_solution = temp_solution
                else:
                    continue
            '''
            if existing_occupation:
                temp_solution = copy.deepcopy(solution)
                if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                    if temp_solution.cost < best_cost:
                        best_cost = temp_solution.cost
                        best_berth = berth
                        best_solution = temp_solution
            else:
                # 不存在占用，创建子集合SBZY
                sub_occupations = []
                for occ in solution.berth_sequences[berth]:
                    if occ['start'] >= time_threshold and occ['end'] < time_threshold_S:
                        sub_occupations.append(occ)

                # if not sub_occupations:
                # continue

                # 遍历所有可能的位置索引 (0: 第一个元素之前, 1: 第一个和第二个之间, ..., n: 最后一个元素之后)
                n = len(sub_occupations)
                for pos_index in range(n + 1):
                    # 创建临时解
                    temp_solution = copy.deepcopy(solution)

                    # 创建新的占用
                    if self.create_new_occupation(temp_solution, vessel, berth, pos_index, sub_occupations,
                                                  time_threshold, time_threshold_S, task_type, workload):
                        # 尝试插入
                        if self.try_insert_task(temp_solution, vessel, task_type, workload, berth):
                            if temp_solution.cost < best_cost:
                                best_cost = temp_solution.cost
                                best_berth = berth
                                best_solution = temp_solution

        # 如果有可行解，更新原始解
        if best_berth is not None:
            solution.__dict__.update(best_solution.__dict__)
            return True

        return False

    def try_insert_task(self, solution, vessel, task_type, workload, berth):
        """尝试将任务插入到指定泊位（支持并行任务）"""
        instance = self.instance

        # 输出解的所有详细信息
        # print("1---------------------")
        # print_solution_details(solution)
        # print("1---------------------")

        # 获取船舶的预计到达时间
        ship_data = next((s for s in instance.ships if s['vessel'] == vessel), None)
        if ship_data:
            tva = ship_data['tva']
        else:
            tva = 0

        # 步骤1: 设置时间阈值TY - 获取泊位b上船舶v占用的开始时间
        TY = 0
        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] == vessel:
                TY = max(tva, occ['start'])
                break

        # 新增代码: 计算时间阈值TYZ
        # 获取船舶v的移泊序列
        vessel_movements = solution.vessel_movements.get(vessel, [])

        # 查找当前泊位在移泊序列中的位置
        current_berth_index = -1
        for i, b in enumerate(vessel_movements):
            if b == berth:
                current_berth_index = i
                break

        # 计算TYZ
        if current_berth_index == 0:  # 当前泊位是移泊序列中的第一个
            # TYZ = solution.berth_occupation_start.get((vessel, berth), 0)
            TYZ = TY
            # 获取船舶v的预计到达时间
            # ship_data = next((s for s in self.instance.ships if s['vessel'] == vessel), None)
            # if ship_data:
            # TYZ = ship_data['tva']
            # else:
            # TYZ = 0  # 默认值，理论上不应该发生
        elif current_berth_index > 0:  # 当前泊位不是第一个
            # 获取上一个泊位
            previous_berth = vessel_movements[current_berth_index - 1]
            # 获取上一个泊位的结束时间
            previous_end_time = solution.berth_occupation_end.get((vessel, previous_berth), 0)
            # 计算TYZ = 上一个泊位结束时间 + 移泊时间
            TYZ = previous_end_time + self.instance.tvc
        else:  # 当前泊位不在移泊序列中
            # 这种情况理论上不应该发生，但为了安全起见，设置默认值
            TYZ = 0

        # 步骤2: 调用重新调度函数
        precedence_start = self.get_time_threshold(solution, vessel, task_type)
        original_start_time = max(TY, TYZ, precedence_start)
        self.reschedule_after_insertion(solution, vessel, task_type, berth, workload, original_start_time)

        # 输出解的所有详细信息
        # print("3---------------------")
        # print_solution_details(solution)
        # print("3---------------------")

        # 步骤4: 更新船舶结束时间
        # vessel_end = solution.vessel_end.get(vessel, 0)
        # tvd = next(ship['tvd'] for ship in instance.ships if ship['vessel'] == vessel)
        # solution.delay[vessel] = max(0, vessel_end - tvd)

        # 步骤4: 更新船舶结束时间和延迟
        self.update_vessel_end_times_and_delays(solution)
        '''
        # 获取船舶在所有泊位上的结束时间，取最大值作为船舶结束时间
        vessel_end_time = 0.0
        for b in self.instance.berths:
            if (vessel, b) in solution.berth_occupation_end:
                end_time = solution.berth_occupation_end[(vessel, b)]
                if end_time > vessel_end_time:
                    vessel_end_time = end_time

        # 更新船舶结束时间
        solution.vessel_end[vessel] = vessel_end_time

        # 计算船舶延迟
        tvd = next(ship['tvd'] for ship in self.instance.ships if ship['vessel'] == vessel)
        solution.delay[vessel] = max(0, vessel_end_time - tvd)
        '''

        # 步骤5: 重新计算总成本
        solution.cost = solution.calculate_cost()

        # 步骤3: 检查优先级约束
        priority_edges = instance.get_priority_constraints(vessel)
        for (j_prime, j) in priority_edges:
            # 如果两个任务都已经调度
            if (vessel, j_prime) in solution.task_end and (vessel, j) in solution.task_start:
                if solution.task_end[(vessel, j_prime)] > solution.task_start[(vessel, j)]:
                    # 优先级约束不满足
                    print(
                        f"插入任务 ({vessel}, {task_type}) 违反优先级约束: {j_prime} 结束于 {solution.task_end[(vessel, j_prime)]}, {j} 开始于 {solution.task_start[(vessel, j)]}")
                    # return False
        # print("2---------------------")

        return True

    def reschedule_after_insertion(self, solution, vessel, task_type, berth, workload, original_start_time):
        """
        场景2：插入任务后的重新调度

        参数:
        solution -- 当前解
        vessel -- 船舶ID
        task_type -- 插入的任务类型
        berth -- 泊位ID
        workload -- 任务工作量
        original_start_time -- 原占用块的开始时间（作为阈值时间）

        返回:
        重新调度后的解
        """
        instance = self.instance

        # 步骤1: 插入的占用块调度
        # 查找船舶在泊位上的占用块

        occupation = None
        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] == vessel:
                occupation = occ
                break

        # if occupation:
        # 存在占用块Kbv
        # 将任务插入到占用块
        occupation['tasks'].append((task_type, workload))

        # 更新泊位分配
        solution.X[(vessel, task_type, berth)] = 1
        solution.task_to_berth[(vessel, task_type)] = berth
        solution.Z[(vessel, berth)] = 1

        # 对占用块进行调度（使用新的岸桥调度逻辑）
        #tasks = occupation['tasks']
        #start_time = occupation['start']
        #start_time1 = original_start_time

        tasks = occupation['tasks']
        occupation_task_types = [task[0] for task in tasks]
        occupation_task_type_set = set(occupation_task_types)
        initial_start_time = original_start_time

        # 使用新的岸桥调度逻辑
        #end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, start_time1)
        end_time = self.schedule_berth_occupation(vessel, berth, tasks, solution, initial_start_time)
        '''
        # 获取当前船舶v在泊位b上占用的下一个船舶占用的开始时间
        next_occupation_start = float('inf')  # 初始化为无穷大，表示没有下一个占用

        # 查找当前占用在泊位序列中的位置
        berth_occupations = solution.berth_sequences[berth]
        current_index = -1
        for i, occ in enumerate(berth_occupations):
            if occ['vessel'] == vessel and occ['start'] == occupation['start']:
                current_index = i
                break

        # 如果找到当前占用并且不是最后一个，获取下一个占用的开始时间
        if current_index >= 0 and current_index < len(berth_occupations) - 1:
            next_occupation_start = berth_occupations[current_index + 1]['start']

        # 将 old_occupation_end_2 设置为下一个船舶占用的开始时间
        old_occupation_start_2 = next_occupation_start
        '''
        '''
        # 更新占用结束时间
        old_occupation_end_2 = occupation['end']
        occupation['end'] = end_time
        occupation['start'] = start_time1
        solution.berth_occupation_end[(vessel, berth)] = end_time
        solution.berth_occupation_start[(vessel, berth)] = start_time1

        # 新增代码: 计算时间阈值SJYZ
        # 获取船舶v的移泊序列
        vessel_movements = solution.vessel_movements.get(vessel, [])

        # 查找当前泊位在移泊序列中的位置
        current_berth_index = -1
        for i, b in enumerate(vessel_movements):
            if b == berth:
                current_berth_index = i
                break

        # 计算SJYZ
        if current_berth_index == len(vessel_movements) - 1:  # 当前泊位是移泊序列中的最后一个
            SJYZ = float('inf')  # 无穷大
        elif current_berth_index >= 0 and current_berth_index < len(vessel_movements) - 1:  # 当前泊位不是最后一个
            # 获取下一个泊位
            next_berth = vessel_movements[current_berth_index + 1]
            # 获取下一个泊位的开始时间
            SJYZ = solution.berth_occupation_start.get((vessel, next_berth), float('inf'))
        else:  # 当前泊位不在移泊序列中
            # 这种情况理论上不应该发生，但为了安全起见，设置为无穷大
            SJYZ = float('inf')

        # b) 判断Evb是否小于等于泊位b上船舶v的下一个船舶占用的开始时间
        next_occupation_start = float('inf')
        for occ in solution.berth_sequences[berth]:
            if occ['vessel'] != vessel and occ['start'] >= old_occupation_end_2:
                next_occupation_start = occ['start']
                break

        if end_time <= next_occupation_start and end_time + self.instance.tvc <= SJYZ:
            # 终止场景2
            return solution
        else:
            # 跳转到步骤2
            return self.global_reschedule_after_insertion(solution, berth, start_time1)
            # else:
            # 不存在占用块Kbv
            # 在泊位b最后创建一个占用块Kbv(new)
            # 获取前一个占用块的结束时间
            '''
        '''
            prev_occupation_end = 0
            if solution.berth_sequences[berth]:
                prev_occupation_end = solution.berth_sequences[berth][-1]['end']

            # 创建新的占用块
            occupation = {
                'vessel': vessel,
                'start': prev_occupation_end,
                'end': prev_occupation_end,
                'tasks': [(task_type, workload)]
            }
            solution.berth_sequences[berth].append(occupation)

            # 更新泊位分配
            solution.X[(vessel, task_type, berth)] = 1
            solution.task_to_berth[(vessel, task_type)] = berth
            solution.Z[(vessel, berth)] = 1
            solution.berth_occupation_start[(vessel, berth)] = prev_occupation_end

            # 更新船舶移泊序列
            if vessel not in solution.vessel_movements:
                solution.vessel_movements[vessel] = []
            if berth not in solution.vessel_movements[vessel]:
                solution.vessel_movements[vessel].append(berth)

            # 判断任务i是否为需要岸桥的任务
            if task_type in instance.JQ:
                # 是需要岸桥的任务
                # 选择最佳岸桥
                best_crane, best_start = self.select_best_crane(solution, prev_occupation_end,
                                                                workload / instance.vj[task_type])

                if best_crane is None:
                    # 如果没有找到合适的岸桥，选择负载最轻的岸桥
                    crane_workloads = {}
                    for crane in instance.Q:
                        total_workload = sum(
                            task['end'] - task['start'] for task in solution.crane_sequences.get(crane, []))
                        crane_workloads[crane] = total_workload

                    best_crane = min(crane_workloads, key=crane_workloads.get)
                    best_start = max(prev_occupation_end,
                                     solution.crane_sequences[best_crane][-1]['end'] if solution.crane_sequences[
                                         best_crane] else prev_occupation_end)

                # (2) 设置任务开始时间
                task_start = best_start
                task_end = task_start + workload / instance.vj[task_type]

                # 记录岸桥分配
                solution.Y[(vessel, task_type, best_crane)] = 1
                solution.task_to_crane[(vessel, task_type)] = best_crane

                # 更新岸桥序列
                new_task = {
                    'vessel': vessel,
                    'task': task_type,
                    'start': task_start,
                    'end': task_end
                }
                if best_crane not in solution.crane_sequences:
                    solution.crane_sequences[best_crane] = []
                solution.crane_sequences[best_crane].append(new_task)
                solution.crane_sequences[best_crane].sort(key=lambda x: x['start'])

                # 更新任务时间
                solution.task_start[(vessel, task_type)] = task_start
                solution.task_end[(vessel, task_type)] = task_end

                # 更新占用结束时间
                occupation['end'] = task_end
                solution.berth_occupation_end[(vessel, berth)] = task_end
            else:
                # 不需要岸桥的任务
                task_start = prev_occupation_end
                task_end = task_start + workload / instance.vj[task_type]

                # 更新任务时间
                solution.task_start[(vessel, task_type)] = task_start
                solution.task_end[(vessel, task_type)] = task_end

                # 更新占用结束时间
                occupation['end'] = task_end
                solution.berth_occupation_end[(vessel, berth)] = task_end

            # 转入步骤2
            return self.global_reschedule_after_insertion(solution, berth, original_start_time)
            '''
        # 更新占用结束时间
        # occupation['end'] = end_time
        #occupation['start'] = start_time1
        #solution.berth_occupation_end[(vessel, berth)] = end_time
        #solution.berth_occupation_start[(vessel, berth)] = start_time1
        scheduled_starts = [
            solution.task_start[(vessel, t_type)]
            for t_type in occupation_task_types
            if (vessel, t_type) in solution.task_start
        ]
        actual_start_time = min(scheduled_starts) if scheduled_starts else initial_start_time

        occupation['end'] = end_time
        occupation['start'] = actual_start_time
        solution.berth_occupation_end[(vessel, berth)] = end_time
        solution.berth_occupation_start[(vessel, berth)] = actual_start_time


        # 调整移泊顺序并按照开始时间排序泊位占用
        self.update_vessel_movement_order(solution, vessel)
        berth_occupations = solution.berth_sequences[berth]
        berth_occupations.sort(key=lambda x: x['start'])

        # 计算前一个和后一个占用的界限
        occ_index = berth_occupations.index(occupation)
        prev_occupation_end = berth_occupations[occ_index - 1]['end'] if occ_index > 0 else float('-inf')
        next_occupation_start = berth_occupations[occ_index + 1]['start'] if occ_index < len(
            berth_occupations) - 1 else float('inf')

        # 新增代码: 计算时间阈值SJYZ
        # 获取船舶v的移泊序列
        vessel_movements = solution.vessel_movements.get(vessel, [])

        # 查找当前泊位在移泊序列中的位置
        current_berth_index = -1
        for i, b in enumerate(vessel_movements):
            if b == berth:
                current_berth_index = i
                break

        # 计算SJYZ
        if current_berth_index == len(vessel_movements) - 1:  # 当前泊位是移泊序列中的最后一个
            SJYZ = float('inf')  # 无穷大
        elif current_berth_index >= 0 and current_berth_index < len(vessel_movements) - 1:  # 当前泊位不是最后一个
            # 获取下一个泊位
            next_berth = vessel_movements[current_berth_index + 1]
            # 获取下一个泊位的开始时间
            SJYZ = solution.berth_occupation_start.get((vessel, next_berth), float('inf'))
        else:  # 当前泊位不在移泊序列中
            # 这种情况理论上不应该发生，但为了安全起见，设置为无穷大
            SJYZ = float('inf')

        #needs_reschedule = False
        violation_threshold = actual_start_time
        needs_reschedule = False

        # a) 检查是否与前一个占用发生冲突
        if actual_start_time < prev_occupation_end - 1e-6:
            needs_reschedule = True

        # b) 判断Evb是否小于等于泊位b上船舶v的下一个船舶占用的开始时间␊
        if end_time > next_occupation_start + 1e-6:
            needs_reschedule = True

        # c) 检查移泊时间约束
        if end_time + self.instance.tvc > SJYZ + 1e-6:
            needs_reschedule = True

        # d) 如果后续占用的开始时间早于当前占用的结束时间，也需要全局重排
        if SJYZ + 1e-6 < end_time:
            needs_reschedule = True


        # d) 检查后续任务是否违反前置约束
        if not needs_reschedule:
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                if j_prime in occupation_task_type_set:
                    successor_start = solution.task_start.get((vessel, j))
                    predecessor_end = solution.task_end.get((vessel, j_prime))
                    if successor_start is not None and predecessor_end is not None and successor_start < predecessor_end - 1e-6:
                        violation_threshold = min(violation_threshold, successor_start)
                        needs_reschedule = True
                        break

        if not needs_reschedule:
            return solution

        # 跳转到步骤2
        return self.global_reschedule_after_insertion(solution, berth, violation_threshold)

    def global_reschedule_after_insertion(self, solution, berth, threshold_time):
        """
        场景2的步骤2：全局重新调度
        """
        instance = self.instance

        # 输出解的所有详细信息
        # print("2---------------------")
        # print_solution_details(solution)
        # print("2---------------------")

        # (1) 重新设置各个资源可用时间
        # 泊位资源重新设置
        berth_availability = {}
        for b in instance.berths:
            # 找到结束时间小于等于阈值时间的最大结束时间
            max_end_time = 0
            for occ in solution.berth_sequences[b]:
                if occ['end'] <= threshold_time and occ['end'] > max_end_time:
                    max_end_time = occ['end']
            berth_availability[b] = max_end_time

        # 岸桥资源重新设置
        crane_availability = {}
        for crane in instance.Q:
            # 找到结束时间小于等于阈值时间的最大结束时间
            max_end_time = 0
            for task in solution.crane_sequences[crane]:
                if task['end'] <= threshold_time and task['end'] > max_end_time:
                    max_end_time = task['end']
            crane_availability[crane] = max_end_time

        # (2) 调度计划初始化
        # 对于结束时间大于阈值时间的占用，初始化时间相关的决策变量和岸桥相关的决策变量
        for b in instance.berths:
            for occ in solution.berth_sequences[b]:
                if occ['end'] > threshold_time:
                    vessel = occ['vessel']
                    # 初始化时间变量
                    for task in occ['tasks']:
                        task_t = task[0]
                        if (vessel, task_t) in solution.task_start:
                            del solution.task_start[(vessel, task_t)]
                        if (vessel, task_t) in solution.task_end:
                            del solution.task_end[(vessel, task_t)]

                    # 初始化岸桥变量
                    for task in occ['tasks']:
                        task_t = task[0]
                        if task_t in instance.JQ and (vessel, task_t) in solution.task_to_crane:
                            crane = solution.task_to_crane[(vessel, task_t)]
                            solution.Y[(vessel, task_t, crane)] = 0
                            # 从岸桥序列中删除任务
                            crane_sequence = solution.crane_sequences[crane]
                            for i, crane_task in enumerate(crane_sequence):
                                if crane_task['vessel'] == vessel and crane_task['task'] == task_t:
                                    del crane_sequence[i]
                                    break
                            del solution.task_to_crane[(vessel, task_t)]

        # (3) 重新调度
        # 步骤1：创建重调度集合CDDS
        CDDS = []
        for b in instance.berths:
            for occ in solution.berth_sequences[b]:
                if occ['end'] > threshold_time:
                    CDDS.append(occ)

        '''
        def sort_occupations(occupation1, occupation2, instance):
            """
            排序函数：首先按开始时间升序排序，对于开始时间相同的占用，
            根据任务优先级关系排序（有优先关系的排在前面）
            """
            # 第一排序准则：开始时间
            if occupation1['start'] < occupation2['start']:
                return -1
            elif occupation1['start'] > occupation2['start']:
                return 1
            else:
                # 开始时间相同，应用第二排序准则：任务优先级
                # 检查occupation1中是否有任务优先于occupation2中的任务
                vessel1 = occupation1['vessel']
                vessel2 = occupation2['vessel']

                # 只有同一船舶的不同占用才可能有优先级关系
                if vessel1 == vessel2:
                    # 获取该船舶的优先级约束
                    priority_edges = instance.get_priority_constraints(vessel1)

                    # 检查occupation1中的任务是否优先于occupation2中的任务
                    for task1 in occupation1['tasks']:
                        task_type1 = task1[0]
                        for task2 in occupation2['tasks']:
                            task_type2 = task2[0]
                            # 如果存在优先级关系 (task_type1, task_type2)
                            if (task_type1, task_type2) in priority_edges:
                                return -1  # occupation1排在前面

                    # 检查occupation2中的任务是否优先于occupation1中的任务
                    for task2 in occupation2['tasks']:
                        task_type2 = task2[0]
                        for task1 in occupation1['tasks']:
                            task_type1 = task1[0]
                            # 如果存在优先级关系 (task_type2, task_type1)
                            if (task_type2, task_type1) in priority_edges:
                                return 1  # occupation2排在前面

                # 如果没有优先级关系或不是同一船舶，保持原顺序
                return 0
        '''

        # 使用自定义排序函数进行排序，传递 instance 参数
        # SCDDS = sorted(CDDS, key=functools.cmp_to_key(lambda x, y: sort_occupations(x, y, instance)))

        # 根据开始时间升序排序
        # SCDDS = sorted(CDDS, key=lambda x: x['start'])
        # SCDDS = sorted(CDDS, key=functools.cmp_to_key(sort_occupations))

        def get_berth_of_occ(occupation):
            for berth_id in instance.berths:
                if occupation in solution.berth_sequences[berth_id]:
                    return berth_id
            return None

        def sort_key(occupation):
            berth_id = get_berth_of_occ(occupation)
            vessel_sequence = solution.vessel_movements.get(occupation['vessel'], [])
            if berth_id in vessel_sequence:
                seq_index = vessel_sequence.index(berth_id)
            else:
                seq_index = 0
            return (seq_index, occupation['start'])

        # 根据船舶移泊顺序和旧的开始时间排序
        SCDDS = sorted(CDDS, key=sort_key)

        # 步骤2：遍历SCDDS中的元素
        for occ in SCDDS:
            vessel = occ['vessel']
            b = None
            # 查找泊位b
            for berth in instance.berths:
                if occ in solution.berth_sequences[berth]:
                    b = berth
                    break

            if b is None:
                continue

            tasks = occ['tasks']

            # 获取上一个船舶占位的结束时间
            prev_occupation_end = berth_availability[b]

            # 获取船舶上一个泊位占位的结束时间+移泊时间
            # 获取船舶上一个泊位占位的结束时间+移泊时间

            berth_sequence = solution.vessel_movements[vessel]

            berth_index = berth_sequence.index(b)

            if berth_index == 0:
                prev_berth_end = self.get_previous_berth_end_time(solution, vessel, b)
            else:
                prev_berth_end = self.get_previous_berth_end_time(solution, vessel, b) + instance.tvc

            '''
            # 考虑跨泊位的优先级约束
            priority_constraint_start = 0
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                # 检查是否有任务在当前占用中，而其前置任务在其他泊位
                j_in_current = any(t[0] == j for t in tasks)
                j_prime_in_other = (vessel, j_prime) in solution.task_end and not any(t[0] == j_prime for t in tasks)

                if j_in_current and j_prime_in_other:
                    # 前置任务在其他泊位，需要等待其完成
                    j_prime_end = solution.task_end[(vessel, j_prime)]
                    if j_prime_end > priority_constraint_start:
                        priority_constraint_start = j_prime_end

            # 计算开始时间（考虑所有约束）
            start_time = max(prev_occupation_end, prev_berth_end, priority_constraint_start)
'''

            # 获取船舶的预计到达时间
            ship_data = next((s for s in self.instance.ships if s['vessel'] == vessel), None)
            if ship_data:
                tva = ship_data['tva']
            else:
                tva = 0  # 默认值，理论上不应该发生

            # 考虑跨泊位的优先级约束
            priority_constraint_start = 0.0
            priority_edges = instance.get_priority_constraints(vessel)
            for (j_prime, j) in priority_edges:
                if any(t[0] == j for t in tasks):
                    if any(t[0] == j_prime for t in tasks):
                        continue

                    predecessor_end = None
                    berth_prime = solution.task_to_berth.get((vessel, j_prime))
                    if berth_prime is not None:
                        predecessor_end = solution.berth_occupation_end.get((vessel, berth_prime))

                    if predecessor_end is None and (vessel, j_prime) in solution.task_end:
                        predecessor_end = solution.task_end[(vessel, j_prime)]

                    if predecessor_end is not None and predecessor_end > priority_constraint_start:
                        priority_constraint_start = predecessor_end

            # 计算开始时间
            start_time = max(prev_occupation_end, prev_berth_end, priority_constraint_start, tva)

            # start_time = max(prev_occupation_end, prev_berth_end)

            # 调度占用
            end_time = self.schedule_berth_occupation(vessel, b, tasks, solution, start_time)

            # 更新占用信息
            occ['start'] = start_time
            occ['end'] = end_time
            solution.berth_occupation_start[(vessel, b)] = start_time
            solution.berth_occupation_end[(vessel, b)] = end_time

            solution.berth_sequences[b].sort(key=lambda x: x['start'])
            self.update_vessel_movement_order(solution, vessel)

            # 更新资源可用时间
            berth_availability[b] = end_time

            # 更新岸桥可用时间（这里简化处理，实际应该从调度过程中获取）
            # 这里需要根据实际调度情况更新crane_availability

        return solution

    # ================== 算子管理 ==================
    def select_destroy_operator(self):
        """根据权重选择破坏算子"""
        total_weight = sum(self.destroy_weights)
        if total_weight == 0:
            return random.choice(range(len(self.destroy_operators)))

        probs = [w / total_weight for w in self.destroy_weights]
        return np.random.choice(len(self.destroy_operators), p=probs)

    def select_repair_operator(self):
        """根据权重选择修复算子"""
        total_weight = sum(self.repair_weights)
        if total_weight == 0:
            return random.choice(range(len(self.repair_operators)))

        probs = [w / total_weight for w in self.repair_weights]
        return np.random.choice(len(self.repair_operators), p=probs)

    def update_operator_weights(self, iter_count):
        """更新算子权重"""
        if iter_count % self.update_freq == 0:
            # 更新破坏算子权重
            for i in range(len(self.destroy_operators)):
                if self.destroy_counts[i] > 0:
                    avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                    self.destroy_weights[i] = (self.lambda_ * self.destroy_weights[i] +
                                               (1 - self.lambda_) * avg_score)
                self.destroy_scores[i] = 0
                self.destroy_counts[i] = 0

            # 更新修复算子权重
            for i in range(len(self.repair_operators)):
                if self.repair_counts[i] > 0:
                    avg_score = self.repair_scores[i] / self.repair_counts[i]
                    self.repair_weights[i] = (self.lambda_ * self.repair_weights[i] +
                                              (1 - self.lambda_) * avg_score)
                self.repair_scores[i] = 0
                self.repair_counts[i] = 0

    # ================== 接受准则 ==================
    def acceptance_probability(self, current_cost, new_cost, temp):
        """计算接受新解的概率（模拟退火）"""
        if new_cost < current_cost:
            return 1.0
        else:
            return math.exp((current_cost - new_cost) / temp)

    # ================== 多样性保持 ==================
    def apply_diversification(self, solution, iter_count, no_improve_count):
        """应用多样性保持策略"""
        # 重启机制：如果长时间无改进
        if no_improve_count >= self.diversity_freq:
            # 基于当前最优解扰动
            new_solution = copy.deepcopy(self.best_solution)

            # 随机移除一些任务
            num_to_remove = min(5, len(self.instance.ships) * 3)  # 最多移除5个任务
            all_tasks = []
            for ship in self.instance.ships:
                vessel = ship['vessel']
                for task in ship['tasks']:
                    task_type = task[0]
                    all_tasks.append((vessel, task_type))

            if all_tasks:
                tasks_to_remove = random.sample(all_tasks, min(num_to_remove, len(all_tasks)))
                for vessel, task_type in tasks_to_remove:
                    self.remove_task(new_solution, vessel, task_type)

            # 使用随机修复重新插入
            self.random_insertion(new_solution)
            return new_solution

        # 定期扰动
        if iter_count % self.perturb_freq == 0:
            new_solution = copy.deepcopy(solution)

            # 随机移除少量任务
            num_to_remove = min(3, len(self.instance.ships) * 2)  # 最多移除3个任务
            all_tasks = []
            for ship in self.instance.ships:
                vessel = ship['vessel']
                for task in ship['tasks']:
                    task_type = task[0]
                    all_tasks.append((vessel, task_type))

            if all_tasks:
                tasks_to_remove = random.sample(all_tasks, min(num_to_remove, len(all_tasks)))
                for vessel, task_type in tasks_to_remove:
                    self.remove_task(new_solution, vessel, task_type)

            # 使用随机修复重新插入
            self.random_insertion(new_solution)
            return new_solution

        return solution

    # ================== ALNS主循环 ==================
    # ================== ALNS主循环 ==================
    def solve(self):
        """执行ALNS算法求解"""
        start_time = time.time()
        last_output_time = start_time  # 记录上次输出时间

        # 步骤1: 初始化
        self.current_solution = self.construct_initial_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.history.append(self.current_solution.cost)

        temp = self.initial_temp
        no_improve_count = 0

        print(f"初始解成本: {self.current_solution.cost:.2f}")

        # 步骤2: 迭代过程
        for iter_count in range(1, self.max_iter + 1):
            # 检查时间限制
            elapsed = time.time() - start_time
            if elapsed > self.max_time:
                print(f"达到时间限制: {elapsed:.2f}秒")
                break

            # 每20秒输出一次累计运行时间
            current_time = time.time()
            if current_time - last_output_time >= 50:
                print(f"累计运行时间: {elapsed:.2f}秒, 迭代次数: {iter_count}")
                last_output_time = current_time

            # 每50次迭代输出一次当前解信息
            if iter_count % 50 == 0:
                print(f"\n=== 迭代 {iter_count} ===")
                print(f"当前解成本: {self.current_solution.cost:.2f}")
                print(f"最优解成本: {self.best_solution.cost:.2f}")
                print(f"温度: {temp:.2f}")
                print(f"无改进计数: {no_improve_count}")

                # 输出当前解的关键信息
                print("\n当前解关键信息:")
                print("船舶延迟情况:")
                for vessel, delay in self.current_solution.delay.items():
                    if delay > 0:
                        print(f"  船舶 {vessel}: 延迟 {delay:.2f} 小时")

                # 输出破坏和修复算子的使用情况
                print("\n算子使用情况:")
                print("破坏算子权重:", [f"{w:.2f}" for w in self.destroy_weights])
                print("修复算子权重:", [f"{w:.2f}" for w in self.repair_weights])
                print("=" * 50)

            # 选择算子
            destroy_idx = self.select_destroy_operator()
            repair_idx = self.select_repair_operator()

            # 应用破坏算子
            destroyed_solution = self.destroy_operators[destroy_idx](self.current_solution)

            # print("1---------------------")
            # print_solution_details(destroyed_solution)
            # print("1---------------------")

            # 应用修复算子
            new_solution = self.repair_operators[repair_idx](destroyed_solution)

            # print("1---------------------")
            # print_solution_details(new_solution)
            # print("1---------------------")

            # 计算接受概率
            accept_prob = self.acceptance_probability(
                self.current_solution.cost, new_solution.cost, temp
            )

            # 更新算子得分
            if new_solution.cost < self.current_solution.cost:
                # 改进解
                self.destroy_scores[destroy_idx] += self.sigma1
                self.repair_scores[repair_idx] += self.sigma1
                self.current_solution = new_solution

                if new_solution.cost < self.best_solution.cost:
                    self.best_solution = copy.deepcopy(new_solution)
                    no_improve_count = 0
                    print(
                        f"迭代 {iter_count}: 新最优解成本: {self.best_solution.cost:.2f} (破坏: {self.destroy_names[destroy_idx]}, 修复: {self.repair_names[repair_idx]})")
                else:
                    no_improve_count += 1
            elif random.random() < accept_prob:
                # 接受但未改进
                self.destroy_scores[destroy_idx] += self.sigma2
                self.repair_scores[repair_idx] += self.sigma2
                self.current_solution = new_solution
                no_improve_count += 1
            else:
                # 拒绝解
                self.destroy_scores[destroy_idx] += self.sigma3
                self.repair_scores[repair_idx] += self.sigma3
                no_improve_count += 1

            # 更新算子调用计数
            self.destroy_counts[destroy_idx] += 1
            self.repair_counts[repair_idx] += 1

            # 更新算子权重
            self.update_operator_weights(iter_count)

            # 应用多样性保持
            if no_improve_count > 0:
                self.current_solution = self.apply_diversification(
                    self.current_solution, iter_count, no_improve_count
                )

            # 更新温度
            temp = temp * self.cooling_rate
            if temp < self.min_temp:
                temp = self.min_temp

            # 记录历史
            self.history.append(self.current_solution.cost)

            # 检查终止条件
            if no_improve_count >= self.no_improve_max:
                print(f"连续 {no_improve_count} 次迭代无改进，终止搜索")
                break

            # 每100次迭代打印进度
            if iter_count % 100 == 0:
                print(
                    f"迭代 {iter_count}: 当前成本={self.current_solution.cost:.2f}, 最优成本={self.best_solution.cost:.2f}, 温度={temp:.2f}")

        # 算法结束
        elapsed = time.time() - start_time
        print(f"ALNS完成, 总耗时: {elapsed:.2f}秒")
        print(f"最优解成本: {self.best_solution.cost:.2f}")

        # 验证最终解
        if self.best_solution.is_feasible():
            print("最优解满足所有约束条件")
        else:
            print("警告: 最优解不满足所有约束条件")

        return self.best_solution

    # ================== 结果可视化 ==================
    '''
    def plot_search_history(self):
        """绘制搜索历史"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history, 'b-', linewidth=1)
        plt.xlabel('迭代次数')
        plt.ylabel('总延迟成本')
        plt.title('ALNS搜索过程')
        plt.grid(True)

        # 标记最优解
        min_cost = min(self.history)
        min_index = self.history.index(min_cost)
        plt.plot(min_index, min_cost, 'ro', markersize=8)
        plt.annotate(f'最优解: {min_cost:.2f}',
                     xy=(min_index, min_cost),
                     xytext=(min_index + 10, min_cost + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig('alns_search_history.png')
        plt.show()

    def plot_schedule(self, solution):
        """绘制调度甘特图（显示并行任务）"""
        # 创建泊位甘特图
        plt.figure(figsize=(14, 10))

        # 为每个泊位创建一个子图
        berths = list(self.instance.berths.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(berths)))

        for i, berth in enumerate(berths):
            plt.subplot(len(berths), 1, i + 1)
            sequences = solution.berth_sequences[berth]

            # 按开始时间排序
            sequences.sort(key=lambda x: x['start'])

            # 为每个船舶任务创建不同的y位置
            y_positions = {}
            y_counter = 0

            for occ in sequences:
                vessel = occ['vessel']
                start = occ['start']
                end = occ['end']

                # 绘制占用背景条
                plt.barh(y=y_counter, width=end - start, left=start,
                         height=0.6, color='lightgray', alpha=0.3)

                # 为每个任务绘制单独的条
                task_y = y_counter - 0.2
                if len(occ['tasks']) > 0:  # 确保有任务
                    task_height = 0.4 / len(occ['tasks'])
                else:
                    task_height = 0.4  # 如果没有任务，使用默认高度

                for j, task in enumerate(occ['tasks']):
                    task_type, workload = task
                    task_start = solution.task_start.get((vessel, task_type), start)
                    task_end = solution.task_end.get((vessel, task_type), end)

                    # 绘制任务条
                    plt.barh(y=task_y, width=task_end - task_start, left=task_start,
                             height=task_height, color=colors[i], edgecolor='black')

                    # 标注任务类型
                    plt.text(x=task_start + (task_end - task_start) / 2, y=task_y,
                             s=f"{task_type}", va='center', ha='center', fontsize=8)

                    task_y += task_height

                # 标注船舶
                plt.text(x=start, y=y_counter + 0.3, s=f"船舶 {vessel}",
                         va='bottom', ha='left', fontsize=8)

                y_counter += 1

            plt.title(f'泊位 {berth} 调度计划')
            plt.xlabel('时间')
            plt.ylabel('船舶序列')
            plt.yticks(range(len(sequences)), [f"船舶 {occ['vessel']}" for occ in sequences])
            plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('berth_schedule_parallel.png')
        plt.show()

        # 创建岸桥甘特图
        plt.figure(figsize=(14, 8))

        # 为每个岸桥创建一个子图
        cranes = self.instance.Q
        colors = plt.cm.tab20(np.linspace(0, 1, len(cranes)))

        for i, crane in enumerate(cranes):
            plt.subplot(len(cranes), 1, i + 1)

            # 收集该岸桥的所有任务
            crane_tasks = []
            for ship in self.instance.ships:
                vessel = ship['vessel']
                for task in ship['tasks']:
                    task_type = task[0]
                    if task_type in self.instance.JQ and (vessel, task_type) in solution.task_to_crane:
                        if solution.task_to_crane[(vessel, task_type)] == crane:
                            crane_tasks.append({
                                'vessel': vessel,
                                'task': task_type,
                                'start': solution.task_start.get((vessel, task_type), 0),
                                'end': solution.task_end.get((vessel, task_type), 0)
                            })
            # 按开始时间排序
            crane_tasks.sort(key=lambda x: x['start'])

            for j, task in enumerate(crane_tasks):
                start = task['start']
                end = task['end']

                # 绘制任务条
                plt.barh(y=j, width=end - start, left=start,
                         color=colors[i], edgecolor='black')

                # 标注船舶和任务
                plt.text(x=start + (end - start) / 2, y=j,
                         s=f"V{task['vessel']}: {task['task']}",
                         va='center', ha='center', fontsize=8)

            plt.title(f'岸桥 {crane} 调度计划')
            plt.xlabel('时间')
            plt.ylabel('任务序列')
            plt.yticks(range(len(crane_tasks)), [f"船舶 {task['vessel']}" for task in crane_tasks])
            plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('crane_schedule.png')
        plt.show()
'''


# ================== 测试代码 ==================
# ================== 数据读取与处理 ==================
def create_instance_from_excel(excel_path):
    """从Excel文件创建问题实例"""
    df = pd.read_excel(excel_path)

    # 处理船舶数据
    ships = []
    for index, row in df.iterrows():
        vessel_no = row['Vessel No.']
        tva = float(row['Expected arrival time'])
        tvd = float(row['Expected departure time'])
        tasks = []
        if row['Unloading'] > 0:
            tasks.append(('j2', float(row['Unloading'])))
        if row['Loading'] > 0:
            tasks.append(('j3', float(row['Loading'])))
        if row['Refueling'] > 0:
            tasks.append(('j1', float(row['Refueling'])))
        if row['Charging'] > 0:
            tasks.append(('j4', float(row['Charging'])))
        if row['Water-filling'] > 0:
            tasks.append(('j5', float(row['Water-filling'])))
        if row['Adding ash'] > 0:
            tasks.append(('j6', float(row['Adding ash'])))
        if row['Adding mud'] > 0:
            tasks.append(('j7', float(row['Adding mud'])))
        ships.append({
            'vessel': vessel_no,
            'tva': tva,
            'tvd': tvd,
            'tasks': tasks
        })

    # 泊位属性
    berths = {
        'b1': ['j1'],  # 只支持加油
        'b2': ['j4', 'j7'],  # 支持充电和加水
        'b3': ['j2', 'j3', 'j7'],  # 支持卸货、装货和加泥
        'b4': ['j2', 'j5', 'j6'],  # 支持装货、加水和加灰（支持并行）
        'b5': ['j2', 'j3', 'j4'],  # 支持卸货、装货和加泥
        'b6': ['j3', 'j5', 'j6']  # 支持装货、加水和加灰（支持并行）
    }

    # 构建Bj集合
    Bj = {j: [] for j in ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']}
    for b in berths:
        for j in berths[b]:
            Bj[j].append(b)

    # 岸桥集合
    Q = ['q1', 'q2', 'q3', 'q4']

    # 参数配置
    tvc = 0.5  # 移泊时间（小时）

    vj = {
        'j1': 100,
        'j2': 20,
        'j3': 20,
        'j4': 70,
        'j5': 45,
        'j6': 150,
        'j7': 100,
    }
    M = 1e4  # 大M值

    return ProblemInstance(ships, berths, Bj, Q, tvc, vj, M)


# ================== 解详细信息输出 ==================
def print_solution_details(solution):
    """输出解的所有详细信息"""
    print("=" * 80)
    print("解详细信息")
    print("=" * 80)

    # 1. 输出决策变量
    print("\n1. 决策变量:")
    print("-" * 40)

    # 泊位分配 X
    print("\n泊位分配 X (vessel, task, berth -> 0/1):")
    for key, value in solution.X.items():
        if value == 1:
            print(f"  X{key} = {value}")

    # 岸桥分配 Y
    print("\n岸桥分配 Y (vessel, task, crane -> 0/1):")
    for key, value in solution.Y.items():
        if value == 1:
            print(f"  Y{key} = {value}")

    # 泊位顺序 F
    print("\n泊位顺序 F (vessel, vessel_prime, berth -> 0/1):")
    for key, value in solution.F.items():
        if value == 1:
            print(f"  F{key} = {value}")

    # 岸桥任务顺序 G
    print("\n岸桥任务顺序 G (vessel, vessel_prime, crane, task, task_prime -> 0/1):")
    for key, value in solution.G.items():
        if value == 1:
            print(f"  G{key} = {value}")

    # 移泊顺序 O
    print("\n移泊顺序 O (vessel, berth, berth_prime -> 0/1):")
    for key, value in solution.O.items():
        if value == 1:
            print(f"  O{key} = {value}")

    # 船舶使用泊位 Z
    print("\n船舶使用泊位 Z (vessel, berth -> 0/1):")
    for key, value in solution.Z.items():
        if value == 1:
            print(f"  Z{key} = {value}")

    # 2. 输出时间变量
    print("\n\n2. 时间变量:")
    print("-" * 40)

    # 任务开始时间
    print("\n任务开始时间 T^s (vessel, task -> time):")
    for key, value in solution.task_start.items():
        print(f"  T^s{key} = {value:.2f}")

    # 任务结束时间
    print("\n任务结束时间 T^e (vessel, task -> time):")
    for key, value in solution.task_end.items():
        print(f"  T^e{key} = {value:.2f}")

    # 船舶结束时间
    print("\n船舶结束时间 T^e_v (vessel -> time):")
    for key, value in solution.vessel_end.items():
        print(f"  T^e_v({key}) = {value:.2f}")

    # 泊位占用开始时间
    print("\n泊位占用开始时间 S (vessel, berth -> time):")
    for key, value in solution.berth_occupation_start.items():
        print(f"  S{key} = {value:.2f}")

    # 泊位占用结束时间
    print("\n泊位占用结束时间 E (vessel, berth -> time):")
    for key, value in solution.berth_occupation_end.items():
        print(f"  E{key} = {value:.2f}")

    # 船舶延迟时间
    print("\n船舶延迟时间 D (vessel -> time):")
    for key, value in solution.delay.items():
        print(f"  D({key}) = {value:.2f}")

    # 3. 输出序列变量
    print("\n\n3. 序列变量:")
    print("-" * 40)

    # 泊位占用序列
    print("\n泊位占用序列 σ_b (berth -> list of occupations):")
    for berth, occupations in solution.berth_sequences.items():
        print(f"\n泊位 {berth}:")
        for occ in occupations:
            print(f"  船舶 {occ['vessel']}: [{occ['start']:.2f} - {occ['end']:.2f}]")

    # 岸桥占用序列
    print("\n岸桥占用序列 (crane -> list of tasks):")
    for crane, tasks in solution.crane_sequences.items():
        print(f"\n岸桥 {crane}:")
        for task in tasks:
            print(f"  船舶 {task['vessel']} 任务 {task['task']}: [{task['start']:.2f} - {task['end']:.2f}]")

    # 船舶移泊序列
    print("\n船舶移泊序列 ρ_v (vessel -> list of berths):")
    for vessel, berths in solution.vessel_movements.items():
        print(f"  船舶 {vessel}: {berths}")

    # 4. 输出辅助数据结构
    print("\n\n4. 辅助数据结构:")
    print("-" * 40)

    # 任务到泊位的映射
    print("\n任务到泊位的映射 (vessel, task -> berth):")
    for key, value in solution.task_to_berth.items():
        print(f"  ({key[0]}, {key[1]}) -> {value}")

    # 任务到岸桥的映射
    print("\n任务到岸桥的映射 (vessel, task -> crane):")
    for key, value in solution.task_to_crane.items():
        print(f"  ({key[0]}, {key[1]}) -> {value}")

    print("=" * 80)
    print("解详细信息输出完成")
    print("=" * 80)


# ================== 测试代码修改 ==================
def test_full_alns_with_excel_data():
    """测试完整ALNS算法（使用Excel数据）"""
    print("=" * 50)
    print("测试完整ALNS算法（使用Excel数据）")
    print("=" * 50)

    # 从Excel文件创建实例
    excel_path = r"D:\0 博士阶段资料\毕业资料\Data_2.xlsx"
    instance = create_instance_from_excel(excel_path)

    solver = ALNSSolver(instance, max_iter=100, max_time=600, no_improve_max=100)
    best_solution = solver.solve()

    print("\n最优解详细信息:")
    print(f"总延迟: {best_solution.cost:.2f}")

    # 验证解可行性
    if best_solution.is_feasible():
        print("最优解满足所有约束条件")
    else:
        print("警告: 最优解不满足所有约束条件")

    # 输出解的所有详细信息
    print_solution_details(best_solution)

    # 绘制结果
    # solver.plot_search_history()
    # solver.plot_schedule(best_solution)

    print("=" * 50)
    print("ALNS算法测试完成")
    print("=" * 50)


# ================== 主程序修改 ==================
if __name__ == "__main__":
    # 运行测试（使用Excel数据）
    test_full_alns_with_excel_data()
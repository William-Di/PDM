import numpy as np

# 初始人群
N = 100000
next_id = N
population = np.zeros((N, 11), dtype=np.float32)

# 列索引
IDX_Id = 0
IDX_Sex = 1
IDX_Age = 2
IDX_Vaccine = 3
IDX_Health = 4
IDX_Single = 5
IDX_Cost = 6
IDX_Qaly = 7
IDX_RelationshipDuration = 8  # 伴侣关系剩余年数
IDX_CooldownDuration = 9      # 冷静期剩余年数
IDX_Transition = 0

# 初始化属性
population[:, IDX_Id] = np.arange(N)
population[:, IDX_Sex] = np.random.randint(0, 2, N)
population[:, IDX_Age] = 40
population[:, IDX_Vaccine] = np.random.binomial(1, 0.2, N)
population[:, IDX_Health] = 0
population[:, IDX_Single] = 0
population[:, IDX_Cost] = 0.0
population[:, IDX_Qaly] = 0.0
population[:, IDX_RelationshipDuration] = 0
population[:, IDX_CooldownDuration] = 0
population[:, IDX_Transition] = 0

# 配对与性传播
def simulate_pairing(population, pairing_probability=0.5, min_duration=1, max_duration=10):
    males = np.where((population[:, IDX_Sex] == 0) & (population[:, IDX_Single] == 0))[0]
    females = np.where((population[:, IDX_Sex] == 1) & (population[:, IDX_Single] == 0))[0]

    np.random.shuffle(males)
    np.random.shuffle(females)

    num_pairs = int(min(len(males), len(females)) * pairing_probability)
    males = males[:num_pairs]
    females = females[:num_pairs]

    population[males, IDX_Single] = 1
    population[females, IDX_Single] = 1

    infected_males = population[males, IDX_Health] >= 1
    infected_females = population[females, IDX_Health] >= 1

    transmission_prob = 0.2

    # 女传男
    for i, m in enumerate(males):
        if infected_females[i] and population[m, IDX_Health] == 0:
            if population[m, IDX_Vaccine] == 1:
                if np.random.rand() < transmission_prob * (1 - 0.9):
                    population[m, IDX_Health] = 1
                    population[m, IDX_Cost] += 1000
            else:
                if np.random.rand() < transmission_prob:
                    population[m, IDX_Health] = 1
                    population[m, IDX_Cost] += 1000

    # 男传女
    for i, f in enumerate(females):
        if infected_males[i] and population[f, IDX_Health] == 0:
            if population[f, IDX_Vaccine] == 1:
                if np.random.rand() < transmission_prob * (1 - 0.9):
                    population[f, IDX_Health] = 1
                    population[f, IDX_Cost] += 1000
            else:
                if np.random.rand() < transmission_prob:
                    population[f, IDX_Health] = 1
                    population[f, IDX_Cost] += 1000

    durations = np.random.randint(min_duration, max_duration + 1, size=num_pairs)
    population[males, IDX_RelationshipDuration] = durations
    population[females, IDX_RelationshipDuration] = durations

def update_relationships(population):
    # 有伴侣者关系时长每年减1
    partnered = np.where(population[:, IDX_Single] == 1)[0]
    population[partnered, IDX_RelationshipDuration] -= 1

    # 关系结束，分手的人转变为冷静期状态，冷静期为1年至5年
    breakups = partnered[population[partnered, IDX_RelationshipDuration] <= 0]
    population[breakups, IDX_Single] = 2
    population[breakups, IDX_RelationshipDuration] = 0
    population[breakups, IDX_CooldownDuration] = np.random.randint(1, 5, size=len(breakups))  # 冷静期1年

    # 冷静期每年减1
    cooling = np.where(population[:, IDX_Single] == 2)[0]
    population[cooling, IDX_CooldownDuration] -= 1

    # 冷静期时长为0时，冷静期结束，变为待配对状态
    cooled_off = cooling[population[cooling, IDX_CooldownDuration] <= 0]
    population[cooled_off, IDX_Single] = 0
    population[cooled_off, IDX_CooldownDuration] = 0

# 状态转移，并记录成本和QALY
def state_transition(population):

    # 概率参数
    Probs = {
        "P_h_ci": 0.01,
        "P_ci_h": 0.8,
        "P_ci_pi": 0.1,
        "P_pi_h": 0.3,
        "P_cervical_pi_lc": 0.3,
        "P_oropharyngeal_pi_lc": 0.1,
        "P_penile_pi_lc": 0.1,
        "P_anal_pi_lc": 0.1,
        "P_cervical_lc_rc": 0.3,
        "P_cervical_rc_dc": 0.5,
        "P_oropharyngeal_lc_rc": 0.3,
        "P_oropharyngeal_rc_dc": 0.5,
        "P_penile_lc_rc": 0.3,
        "P_penile_rc_dc": 0.5,
        "P_anal_lc_rc": 0.3,
        "P_anal_rc_dc": 0.5,
        "P_death_h": 0.07,
        "P_death_ci": 0.07,
        "P_death_pi": 0.1,
        "P_death_cervical_lc": 0.2,
        "P_death_cervical_rc": 0.3,
        "P_death_cervical_dc": 0.5,
        "P_death_oropharyngeal_lc": 0.2,
        "P_death_oropharyngeal_rc": 0.3,
        "P_death_oropharyngeal_dc": 0.5,
        "P_death_penile_lc": 0.2,
        "P_death_penile_rc": 0.3,
        "P_death_penile_dc": 0.5,
        "P_death_anal_lc": 0.2,
        "P_death_anal_rc": 0.3,
        "P_death_anal_dc": 0.5,
        "vaccine_efficacy": 0.9
    }

    # 成本参数
    Costs = {
        'cost_health': 0,
        'cost_diagnose': 1000,
        'cost_ci': 200,
        'cost_pi': 2000,
        'cost_cervical_lc': 30000,
        'cost_cervical_rc': 50000,
        'cost_cervical_dc': 80000,
        'cost_oropharyngeal_lc': 30000,
        'cost_oropharyngeal_rc': 50000,
        'cost_oropharyngeal_dc': 80000,
        'cost_penile_lc': 30000,
        'cost_penile_rc': 50000,
        'cost_penile_dc': 80000,
        'cost_anal_lc': 30000,
        'cost_anal_rc': 50000,
        'cost_anal_dc': 80000
    }

    # QALY参数
    Qalys = {
        'qaly_h': 1,
        'qaly_ci': 0.9,
        'qaly_pi': 0.8,
        'qaly_cervical_lc': 0.6,
        'qaly_cervical_rc': 0.5,
        'qaly_cervical_dc': 0.4,
        'qaly_oropharyngeal_lc': 0.6,
        'qaly_oropharyngeal_rc': 0.5,
        'qaly_oropharyngeal_dc': 0.4,
        'qaly_penile_lc': 0.6,
        'qaly_penile_rc': 0.5,
        'qaly_penile_dc': 0.4,
        'qaly_anal_lc': 0.6,
        'qaly_anal_rc': 0.5,
        'qaly_anal_dc': 0.4,
        'qaly_death': 0
    }

    # 重置转移标记，0表示未转移，1表示已转移
    def reset_transition(population):
        population[:, IDX_Transition] = 0

    def transition(population, indices, prob, to_state, cost=0, qaly=0):
        # 每年重置转移状态为0
        reset_transition(population)
        # 只处理未转移个体
        valid_indices = indices[population[indices, IDX_Transition] == 0]
        if len(valid_indices) == 0:
            return
        # 按概率决定哪些个体转移
        rand_vals = np.random.rand(len(valid_indices))
        transitioned = valid_indices[rand_vals < prob]
        if len(transitioned) == 0:
            return
        # 状态更新，实现成本和QALY累加
        population[transitioned, IDX_Health] = to_state
        population[transitioned, IDX_Cost] += cost
        population[transitioned, IDX_Qaly] += qaly
        # 标记为已转移
        population[transitioned, IDX_Transition] = 1

    #健康接种疫苗人群
    hv = np.where((population[:, IDX_Health] == 0) & (population[:, IDX_Vaccine] == 1))[0]
    transition(population, hv, (1 - Probs["P_h_ci"] * (1 - Probs["vaccine_efficacy"]) - Probs["P_death_h"]), 0, Costs['cost_health'], Qalys['qaly_h'])
    transition(population, hv, Probs["P_h_ci"] * (1 - Probs["vaccine_efficacy"]), 1, Costs['cost_diagnose'], Qalys['qaly_ci'])
    transition(population, hv, Probs["P_death_h"], 15, Costs['cost_health'], Qalys['qaly_death'])

    # 健康未接种疫苗人群
    hnv = np.where((population[:, IDX_Health] == 0) & (population[:, IDX_Vaccine] == 0))[0]
    transition(population, hnv, (1 - Probs["P_h_ci"] - Probs["P_death_h"]), 0, Costs['cost_health'], Qalys['qaly_h'])
    transition(population, hnv, Probs["P_h_ci"], 1, Costs['cost_diagnose'], Qalys['qaly_ci'])
    transition(population, hnv, Probs["P_death_h"], 15, Costs['cost_health'], Qalys['qaly_death'])

    #普通感染人群
    ci = np.where(population[:, IDX_Health] == 1)[0]
    transition(population, ci, Probs["P_ci_h"], 0, Costs['cost_ci'], Qalys['qaly_h'])
    transition(population, ci, (1 - Probs["P_ci_h"] - Probs["P_ci_pi"] - Probs["P_death_ci"]), 1, Costs['cost_ci'], Qalys['qaly_ci'])
    transition(population, ci, Probs["P_ci_pi"], 2, Costs['cost_ci'], Qalys['qaly_pi'])
    transition(population, ci, Probs["P_death_ci"], 15, Costs['cost_ci'], Qalys['qaly_death'])

    #持续感染女性
    pi_f = np.where((population[:, IDX_Health] == 2) & (population[:, IDX_Sex] == 1))[0]
    transition(population, pi_f, Probs["P_pi_h"], 0, Costs['cost_pi'], Qalys['qaly_h'])
    transition(population, pi_f, (1 - Probs["P_pi_h"] - Probs["P_cervical_pi_lc"] - Probs["P_death_pi"]), 2, Costs['cost_pi'], Qalys['qaly_pi'])
    transition(population, pi_f, Probs["P_cervical_pi_lc"], 3, Costs['cost_pi'], Qalys['qaly_cervical_lc'])
    transition(population, pi_f, Probs["P_death_pi"], 15, Costs['cost_pi'], Qalys['qaly_death'])

    # 持续感染男性
    pi_m = np.where((population[:, IDX_Health] == 2) & (population[:, IDX_Sex] == 0))[0]
    transition(population, pi_m, Probs["P_pi_h"], 0, Costs['cost_pi'], Qalys['qaly_h'])
    transition(population, pi_m, (1 - Probs["P_pi_h"] - Probs["P_oropharyngeal_pi_lc"] - Probs["P_penile_pi_lc"] - Probs["P_anal_pi_lc"] - Probs["P_death_pi"]), 2, Costs['cost_pi'], Qalys['qaly_pi'])
    transition(population, pi_m, Probs["P_oropharyngeal_pi_lc"], 6, Costs['cost_pi'], Qalys['qaly_oropharyngeal_lc'])
    transition(population, pi_m, Probs["P_penile_pi_lc"], 9, Costs['cost_pi'], Qalys['qaly_penile_lc'])
    transition(population, pi_m, Probs["P_anal_pi_lc"], 12, Costs['cost_pi'], Qalys['qaly_anal_lc'])
    transition(population, pi_m, Probs["P_death_pi"], 15, Costs['cost_pi'], Qalys['qaly_death'])

    #宫颈癌lc女性
    cervical_lc = np.where((population[:, IDX_Health] == 3) & (population[:, IDX_Sex] == 1))[0]
    transition(population, cervical_lc, (1 - Probs["P_cervical_lc_rc"] - Probs["P_death_cervical_lc"]), 3, Costs['cost_cervical_lc'], Qalys['qaly_cervical_lc'])
    transition(population, cervical_lc, Probs["P_cervical_lc_rc"], 4, Costs['cost_cervical_lc'], Qalys['qaly_cervical_rc'])
    transition(population, cervical_lc, Probs["P_death_cervical_lc"], 15, Costs['cost_cervical_lc'], Qalys['qaly_death'])

    # 宫颈癌rc女性
    cervical_rc = np.where((population[:, IDX_Health] == 4) & (population[:, IDX_Sex] == 1))[0]
    transition(population, cervical_rc, (1 - Probs["P_cervical_rc_dc"] - Probs["P_death_cervical_rc"]), 4, Costs['cost_cervical_rc'], Qalys['qaly_cervical_rc'])
    transition(population, cervical_rc, Probs["P_cervical_rc_dc"], 5, Costs['cost_cervical_rc'], Qalys['qaly_cervical_dc'])
    transition(population, cervical_rc, Probs["P_death_cervical_rc"], 15, Costs['cost_cervical_rc'], Qalys['qaly_death'])

    # 宫颈癌dc女性
    cervical_dc = np.where((population[:, IDX_Health] == 5) & (population[:, IDX_Sex] == 1))[0]
    transition(population, cervical_dc, (1 - Probs["P_death_cervical_dc"]), 5, Costs['cost_cervical_dc'], Qalys['qaly_cervical_dc'])
    transition(population, cervical_dc, Probs["P_death_cervical_dc"], 15, Costs['cost_cervical_dc'], Qalys['qaly_death'])

    # 口咽癌lc男性
    oropharyngeal_lc = np.where((population[:, IDX_Health] == 6) & (population[:, IDX_Sex] == 0))[0]
    transition(population, oropharyngeal_lc, (1 - Probs["P_oropharyngeal_lc_rc"] - Probs["P_death_oropharyngeal_lc"]), 6, Costs['cost_oropharyngeal_lc'], Qalys['qaly_oropharyngeal_lc'])
    transition(population, oropharyngeal_lc, Probs["P_oropharyngeal_lc_rc"], 7, Costs['cost_oropharyngeal_lc'], Qalys['qaly_oropharyngeal_rc'])
    transition(population, oropharyngeal_lc, Probs["P_death_oropharyngeal_lc"], 15, Costs['cost_oropharyngeal_lc'], Qalys['qaly_death'])

    # 口咽癌rc男性
    oropharyngeal_rc = np.where((population[:, IDX_Health] == 7) & (population[:, IDX_Sex] == 0))[0]
    transition(population, oropharyngeal_rc, (1 - Probs["P_oropharyngeal_rc_dc"] - Probs["P_death_oropharyngeal_rc"]), 7, Costs['cost_oropharyngeal_rc'], Qalys['qaly_oropharyngeal_rc'])
    transition(population, oropharyngeal_rc, Probs["P_oropharyngeal_rc_dc"], 8, Costs['cost_oropharyngeal_rc'], Qalys['qaly_oropharyngeal_dc'])
    transition(population, oropharyngeal_rc, Probs["P_death_oropharyngeal_rc"], 15, Costs['cost_oropharyngeal_rc'], Qalys['qaly_death'])

    # 口咽癌dc男性
    oropharyngeal_dc = np.where((population[:, IDX_Health] == 8) & (population[:, IDX_Sex] == 0))[0]
    transition(population, oropharyngeal_dc, (1 - Probs["P_death_oropharyngeal_dc"]), 8, Costs['cost_oropharyngeal_dc'], Qalys['qaly_oropharyngeal_dc'])
    transition(population, oropharyngeal_dc, Probs["P_death_oropharyngeal_dc"], 15, Costs['cost_oropharyngeal_dc'], Qalys['qaly_death'])

    # 阴茎癌lc男性
    penile_lc = np.where((population[:, IDX_Health] == 9) & (population[:, IDX_Sex] == 0))[0]
    transition(population, penile_lc, (1 - Probs["P_penile_lc_rc"] - Probs["P_death_penile_lc"]), 9, Costs['cost_penile_lc'], Qalys['qaly_penile_lc'])
    transition(population, penile_lc, Probs["P_penile_lc_rc"], 10, Costs['cost_penile_lc'], Qalys['qaly_penile_rc'])
    transition(population, penile_lc, Probs["P_death_penile_lc"], 15, Costs['cost_penile_lc'], Qalys['qaly_death'])

    # 阴茎癌rc男性
    penile_rc = np.where((population[:, IDX_Health] == 10) & (population[:, IDX_Sex] == 0))[0]
    transition(population, penile_rc, (1 - Probs["P_penile_rc_dc"] - Probs["P_death_penile_rc"]), 10, Costs['cost_penile_rc'], Qalys['qaly_penile_rc'])
    transition(population, penile_rc, Probs["P_penile_rc_dc"], 11, Costs['cost_penile_rc'], Qalys['qaly_penile_dc'])
    transition(population, penile_rc, Probs["P_death_penile_rc"], 15, Costs['cost_penile_rc'], Qalys['qaly_death'])

    # 阴茎癌dc男性
    penile_dc = np.where((population[:, IDX_Health] == 11) & (population[:, IDX_Sex] == 0))[0]
    transition(population, penile_dc, (1 - Probs["P_death_penile_dc"]), 11, Costs['cost_penile_dc'], Qalys['qaly_penile_dc'])
    transition(population, penile_dc, Probs["P_death_penile_dc"], 15, Costs['cost_penile_dc'], Qalys['qaly_death'])

    # 肛门癌lc男性
    anal_lc = np.where((population[:, IDX_Health] == 12) & (population[:, IDX_Sex] == 0))[0]
    transition(population, anal_lc, (1 - Probs["P_anal_lc_rc"] - Probs["P_death_anal_lc"]), 12, Costs['cost_anal_lc'], Qalys['qaly_anal_lc'])
    transition(population, anal_lc, Probs["P_anal_lc_rc"], 13, Costs['cost_anal_lc'], Qalys['qaly_anal_rc'])
    transition(population, anal_lc, Probs["P_death_anal_lc"], 15, Costs['cost_anal_lc'], Qalys['qaly_death'])

    # 肛门癌rc男性
    anal_rc = np.where((population[:, IDX_Health] == 13) & (population[:, IDX_Sex] == 0))[0]
    transition(population, anal_rc, (1 - Probs["P_anal_rc_dc"] - Probs["P_death_anal_rc"]), 13, Costs['cost_anal_rc'], Qalys['qaly_anal_rc'])
    transition(population, anal_rc, Probs["P_anal_rc_dc"], 14, Costs['cost_anal_rc'], Qalys['qaly_anal_dc'])
    transition(population, anal_rc, Probs["P_death_anal_rc"], 15, Costs['cost_anal_rc'], Qalys['qaly_death'])

    # 肛门癌dc男性
    anal_dc = np.where((population[:, IDX_Health] == 14) & (population[:, IDX_Sex] == 0))[0]
    transition(population, anal_dc, (1 - Probs["P_death_anal_dc"]), 14, Costs['cost_anal_dc'], Qalys['qaly_anal_dc'])
    transition(population, anal_dc, Probs["P_death_anal_dc"], 15, Costs['cost_anal_dc'], Qalys['qaly_death'])

    return population

# 统计健康状态人数
def count_health_states(population):
    counts = {}
    states = {
        0: "Healthy",
        1: "Ci",  # 轻度感染
        2: "Pi",  # 持续感染
        3: "Cervical_lc",  # 宫颈局部癌
        4: "Cervical_rc",  # 宫颈区域癌
        5: "Cervical_dc",  # 宫颈远端癌
        6: "Oropharyngeal_lc",  # 口咽局部癌
        7: "Oropharyngeal_rc",  # 口咽区域癌
        8: "Oropharyngeal_dc",  # 口咽远端癌
        9: "Penile_lc",  # 阴茎局部癌
        10: "Penile_rc",  # 阴茎区域癌
        11: "Penile_dc",  # 阴茎远端癌
        12: "Anal_lc",  # 肛门局部癌
        13: "Anal_rc",  # 肛门区域癌
        14: "Anal_dc",  # 肛门远端癌
        15: "Death" # 死亡
    }
    for state_code, state_name in states.items():
        counts[state_name] = np.sum(population[:, IDX_Health] == state_code)
    return counts

# 主模拟循环
years = 10
for year in range(years):
    print(f"\n=== Year {year+1} ===")

    # 年龄增加1岁
    population[:, IDX_Age] += 1

    # 关系更新（时长减小，处理分手和冷静）
    update_relationships(population)

    # 配对（只有待配对状态才参与）
    simulate_pairing(population, pairing_probability=0.5)

    # 状态转移（包含成本和QALY累计）
    population = state_transition(population)

    # 统计当年死亡个体数
    deaths = np.sum(population[:, IDX_Health] == 15)

    # 移除死亡个体
    population = population[population[:, IDX_Health] != 15]

    # 汇总统计
    counts = count_health_states(population)
    total_cost = np.sum(population[:, IDX_Cost])
    total_qaly = np.sum(population[:, IDX_Qaly])

    # 输出统计信息
    print("Single states count:",
          np.bincount(population[:, IDX_Single].astype(int), minlength=3))
    print(f"Total Population: {len(population)}")
    print("Health state counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"Deaths this year: {deaths}")
    print(f"Total Cost so far: {total_cost:.2f}")
    print(f"Total QALY so far: {total_qaly:.2f}")

    # 新增40岁新人口，数量为原始N的10%
    new_count = int(N * 0.1)
    new_population = np.zeros((new_count, 11), dtype=np.float32)
    new_population[:, IDX_Id] = np.arange(next_id, next_id + new_count)
    new_population[:, IDX_Sex] = np.random.randint(0, 2, new_count)
    new_population[:, IDX_Age] = 40
    new_population[:, IDX_Vaccine] = np.random.binomial(1, 0.2, new_count)
    new_population[:, IDX_Health] = 0
    new_population[:, IDX_Single] = 0
    new_population[:, IDX_Cost] = 0.0
    new_population[:, IDX_Qaly] = 0.0
    new_population[:, IDX_RelationshipDuration] = 0
    new_population[:, IDX_CooldownDuration] = 0
    new_population[:, IDX_Transition] = 0

    next_id += new_count

    population = np.vstack((population, new_population))

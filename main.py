import os
import sys
import math
import random
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN, MOUSEWHEEL, K_SPACE, K_c, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8
import scipy.stats as stats
import collections
import time
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

################################################################################
# 상세 설명
#
# 본 시뮬레이터는 MEC(Multi-access Edge Computing) 환경을 메타버스적 관점에서 재현하는 종합 모델링 및 시각화 플랫폼입니다.
# 도시는 2D 격자 형태로 구성되며, Isometric 투영을 통해 건물, 나무, 기지국, 엣지 서버 등의 배치를 3D-ish하게 표현합니다.
# 사용자(노드) 이동 경로, 엣지-클라우드 연결 상태, 네트워크 대역폭 변화, 자원할당 및 가격동학 등을 시각화하고,
# Edge 서버와 Cloud 간의 동적 자원 할당/가격 재계산, 에너지 소비와 로드상태 변화를 종합적으로 고려합니다.
#
# 핵심 특징:
# 1. 다차원 도시 모델 및 ISO 투영 시각화
#    - 고층건물, 나무, 기지국, 엣지 서버 배치를 통해 도시 환경 재현
#    - ISO 투영 기반 좌표변환을 통해 2D 그리드 상의 객체를 3D 비주얼로 표현
#
# 2. MEC 및 Cloud 환경 자원 동적 할당 메커니즘
#    - Edge 서버에 대한 자원 증가 시 비용함수(cost_function), 수요-가격 탄력성(demand_price) 모델 반영
#    - 클라우드는 대용량 큐(CLOUD_MAX_Q)를 갖춘 백엔드로 모델링
#    - 사용자들은 Local, Edge, Cloud 선택 시 성능/비용/유틸리티를 기반으로 합리적인 의사결정 수행
#    - Federated Learning, Early Exit, Personalization, Split Computing 등 고도화된 정책 고려 가능
#
# 3. 물리 계층 모델링
#    - Rayleigh 페이딩(Rayleigh fading), Path Loss, Shadowing 등 실제 무선통신 환경에서 발생하는 링크 품질 변동 요소를 반영
#    - 통신기술( LTE, 5G )별 링크 특성 및 페이딩 패턴 모사
#    - 링크 거리(dist), 페이딩, 섀도잉 값에 따라 동적으로 Down/Uplink 속도가 변하며, 시간 경과에 따라 Fading Update Interval마다 재계산
#    - 이를 통해 단순한 고정 대역폭 가정이 아닌 현실적인 QoS 변동 및 네트워크 품질 저하/개선 양상을 정교하게 재현
#
# 4. 비선형 에너지 하베스팅 모델 및 전력비용 계산
#    - GPU 기반 MEC 서버의 전력비용(POWER_COST_PER_HOUR)을 반영
#    - GPU/CPU/VRAM 등 다양한 자원 요소에 따른 동적 비용 계산
#    - 에너지 효율과 자원 할당 간의 상충관계(trade-off)를 정량적으로 모델링
#
# 5. 실시간 이벤트 로깅 및 대규모 시나리오 확장
#    - 계약 체결/해지, QoS 변화, 로드 증가/감소 등 이벤트를 실시간 로그 기록 및 결과 CSV, 그래프 형태로 저장 가능
#    - 초규모(mega-scale) 시나리오로 확장 용이하며, Federated Learning 기반 정책 업데이트, 사용자 이동성 패턴 변경, 계절별 수요 패턴 변화 등 파라미터 조정 용이
#    - 이로써 다양한 MEC 운영정책, 경제적 가격모델, QoS 보장 전략을 테스트하고 논문화할 수 있는 재현성 높은 실험 환경 확보
#
# 결론:
# 이 시뮬레이터는 메타버스 수준의 복잡한 MEC 운영 환경을 실제 산업현장 근접 수준으로 구현하고 있습니다. Rayleigh 페이딩, Path Loss, Shadowing, 다양한 서비스 요구조건, 
# 동적 가격 재계산 및 자원 관리, Federated Learning 적용 가능성 등을 포괄적으로 반영하고 있습니다.
# 이를 통해 연구자는 미래 MEC 운영정책, 자원 할당 알고리즘, 에너지 효율 개선 방안, 유틸리티 최적화 전략 등을 상세히 분석하고, 
# 다양한 학술 논문 및 기술 보고서를 다량으로 생산할 수 있는 강력한 연구 툴로 활용 가능합니다.
#
################################################################################

PAUSED = False
SHOW_CONNECTION_DETAILS = False
SIMULATION_SPEED = 1  # 기본 1배속

RAYLEIGH_SCALE = 0.5
FADING_UPDATE_INTERVAL = 30
COMM_TECH = ["LTE", "5G"]

PL0 = 40.0
N_EXP = 3.5
SHADOWING_STD = 8.0
D0 = 1.0
BLOCK_REAL_DISTANCE = 100.0

SELECTED_LINK = None
LINK_CLICK_DISTANCE_THRESHOLD = 10.0

OPERATORS = ["SKT", "KT", "LGU+"]
SCREEN_WIDTH = 2300
SCREEN_HEIGHT = 1300
FPS = 15
TOP_PANEL_HEIGHT = 300
CITY_MARGIN = 50
RIGHT_PANEL_WIDTH = 300
MAIN_AREA_WIDTH = SCREEN_WIDTH - RIGHT_PANEL_WIDTH

TOTAL_CORES = 600
EDGE_SERVERS_COUNT = 3
p_max = 0.05
beta = 0.00001
k3 = 1.48e-5
k2 = 3.78e-5
k1 = 6.72e-5
k0 = 1.33e-2
r = 0.9

service_mean = 10
service_sd = 2
service_low = 5
service_high = 20

GPU_PER_UNIT = 0.1
VRAM_PER_UNIT = 256
CPU_PER_UNIT = 0.1
RAM_PER_UNIT = 256
STORAGE_PER_UNIT = 1

TASK_GPU_WEIGHTS = {
    "AR": 1.0,
    "VR": 1.5,
    "ML": 2.0,
    "Voice": 0.5
}

def get_arrival_rate(hh):
    base_rates = {
        (0, 6): 2.0,
        (6, 9): 10.0,
        (9, 11): 8.0,
        (11, 14): 9.0,
        (14, 17): 8.0,
        (17, 20): 10.0,
        (20, 24): 6.0
    }
    
    current_rate = 10.0
    for (start, end), rate in base_rates.items():
        if start <= hh < end:
            current_rate = rate
            break
            
    # 시간대 전환 부드럽게
    for (start, end), rate in base_rates.items():
        if abs(hh - start) < 1:
            prev_rate = list(base_rates.values())[(list(base_rates.keys()).index((start, end)) - 1) % len(base_rates)]
            blend_factor = abs(hh - start)
            current_rate = prev_rate * (1 - blend_factor) + rate * blend_factor
            
    return current_rate * random.uniform(0.9, 1.1)

NIGHT_START = 22
NIGHT_END = 6

p_cloud = 0.005
CLOUD_MAX_Q = 1000000
cloud_q = 0

MAX_INITIAL_PEOPLE = 1000
BUILDING_COUNT = 30
TREE_COUNT = 5
NOISE_POINTS = 80

GRID_W = 20
GRID_H = 20
BLOCK_SIZE = 100
ROAD_WIDTH = 20

ISO_SCALE = 0.7
ISO_OFFSETX = 600
ISO_OFFSETY = 800

camera_offset_x = -(GRID_W * BLOCK_SIZE) // 2
camera_offset_y = -(GRID_H * BLOCK_SIZE) // 4
CAMERA_SPEED = 50

DAY_LENGTH = 24.0
NIGHT_CLEANUP_TIME = 12.0

START_HOUR = 13
START_MINUTE = 30

BUILDING_INSIDE_BLOCK = (2, 2)

BASESTATION_COUNT = 5

EDGE_SERVER_MAX_Q = 10000
PRICE_CHECK_INTERVAL = 10
MIN_SPEED_REQUIREMENT = 20.0

pygame.font.init()

def to_iso(x, y):
    x2 = x + camera_offset_x
    y2 = y + camera_offset_y
    iso_x = (x2 - y2) * ISO_SCALE + ISO_OFFSETX
    iso_y = ((x2 + y2) * 0.5) * ISO_SCALE + ISO_OFFSETY
    return (int(iso_x), int(iso_y))

def sample_service_time():
    while True:
        val = np.random.normal(service_mean, service_sd)
        if service_low <= val <= service_high:
            return val * 60.0

def demand_price(q):
    price = p_max - beta * q
    if price < 0:
        price = 0
    return price

def cost_function(q, N):
    return k3 * q ** 3 / (N ** 2) + k2 * (q ** 2) / N + k1 * q + k0 * N

def incremental_cost(Q, q, N):
    return cost_function(Q + q, N) - cost_function(Q, N)

def find_optimal_q_for_N(N):
    best_p = -1e9
    best_q = 0
    for qtest in range(0, 300000, 5000):
        p = demand_price(qtest)
        rev = p * qtest
        c = cost_function(qtest, N)
        prof = rev - c
        if prof > best_p:
            best_p = prof
            best_q = qtest
    p_star = demand_price(best_q)
    return best_q, p_star, best_p

def compute_price(Q, q, N):
    if q <= 0:
        return 0.0
    base_cost = incremental_cost(Q, q, N) / q
    p_target = demand_price(Q + q)
    unit_price = base_cost + max(0, p_target - base_cost) * 0.5
    return unit_price

def bfs_path(start, goal, blocked):
    visited = set()
    queue = deque()
    queue.append((start, None))
    parent = {}
    while queue:
        cur, p = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        parent[cur] = p
        if cur == goal:
            break
        x, y = cur
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if (nx, ny) not in visited and (nx, ny) not in blocked:
                    queue.append(((nx, ny), cur))
    if goal not in parent:
        return [start]
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

OPERATOR_PRICE_RANGES = {
    "SKT": (0.1, 4.0),
    "KT": (0.1, 5.0),
    "LGU+": (0.1, 4.5)
}

def to_musd(price):
    return price * 1000

def from_musd(price):
    return price / 1000

class Menu:
    def __init__(self, options, position, font, bg_color=(240, 240, 240), text_color=(0, 0, 0)):
        self.options = options
        self.position = position
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color
        self.option_height = 30
        self.option_width = 150
        self.visible = True
        self.selected_option = None
        self.rects = []
        self.create_menu_rects()

    def create_menu_rects(self):
        x, y = self.position
        for option in self.options:
            rect = pygame.Rect(x, y, self.option_width, self.option_height)
            self.rects.append(rect)
            y += self.option_height

    def draw(self, screen):
        for idx, option in enumerate(self.options):
            rect = self.rects[idx]
            pygame.draw.rect(screen, self.bg_color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            text_surf = self.font.render(option, True, self.text_color)
            screen.blit(text_surf, (rect.x + 10, rect.y + (self.option_height - text_surf.get_height()) // 2))

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for idx, rect in enumerate(self.rects):
                if rect.collidepoint(mouse_pos):
                    self.selected_option = self.options[idx]
                    self.visible = False
                    return self.selected_option
            self.visible = False
        return None

POWER_COST_PER_HOUR = 20.0

class Person(pygame.sprite.Sprite):
    def __init__(self, gx, gy, color, env, road_nodes, blocked_cells, from_building=False):
        super().__init__()
        self.color = color
        self.env = env
        self.in_contract = False
        self.in_cloud_contract = False
        
        self.req_t = sample_service_time()
        
        self.task_type = random.choice(["AR", "VR", "ML", "Voice"])
        if self.task_type == "AR":
            self.base_quality = 10
        elif self.task_type == "VR":
            self.base_quality = 15
        elif self.task_type == "ML":
            self.base_quality = 20
        else:
            self.base_quality = 8
        
        # Embodied AI: 건물에서 나온 사용자일 경우 품질 향상
        if from_building:
            self.base_quality += 2

        self.alpha = random.uniform(0.005,0.05)
        self.beta = random.uniform(0.005,0.05)
        self.base_utility = random.uniform(5,15)
        self.utility_threshold = 2.0

        base_req = random.randint(10, 100)
        self.req_q = int(base_req * TASK_GPU_WEIGHTS[self.task_type])
        
        self.req_gpu = self.req_q * GPU_PER_UNIT
        self.req_vram = self.req_q * VRAM_PER_UNIT
        self.req_cpu = self.req_q * CPU_PER_UNIT
        self.req_ram = self.req_q * RAM_PER_UNIT
        self.req_storage = self.req_q * STORAGE_PER_UNIT
        
        self.budget = max(2000, np.random.normal(4000, 1000))
        self.current_price = 0
        self.edge_only = (random.random() < 0.05)
        self.alive = True
        self.lifetime = np.random.exponential(5000)
        self.state = "Idle"
        self.link_target = None
        self.remaining_time = self.req_t

        self.blocked_cells = blocked_cells
        self.gx = gx
        self.gy = gy
        self.path = self.select_random_path()
        self.path_idx = 0
        self.speed = 1.0

        self.image = pygame.Surface((12, 12), pygame.SRCALPHA)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()

        self.link_fading_type = None
        self.link_comm_type = None
        self.link_upload_speed = 0.0
        self.link_download_speed = 0.0
        self.fading_update_counter = 0
        self.link_path_loss_db = 0.0
        self.link_shadowing_db = 0.0
        self.rayleigh_factor = 1.0

        self.contract_end_time = None
        if self.in_contract or self.in_cloud_contract:
            self.contract_end_time = self.env.simulation_minutes + self.req_t

        self.from_building = from_building
        
        self.preference_list = random.choice([
            ["EDGE", "CLOUD", "LOCAL"],
            ["CLOUD", "EDGE", "LOCAL"],
            ["CLOUD", "LOCAL", "EDGE"],
            ["EDGE", "LOCAL", "CLOUD"],
            ["LOCAL", "EDGE", "CLOUD"],
            ["LOCAL", "CLOUD", "EDGE"]
        ])
        self.loyalty = random.uniform(0.0,1.0)
        self.consistency_factor = random.uniform(1.0, 2.0)

        # 추가: 어떤 기술 적용 중인지 표시하기 위한 flags
        self.apply_early_exit = False
        self.apply_federated = False
        self.apply_split = False
        self.apply_personalization = False

    def select_random_path(self):
        start = (int(round(self.gx)), int(round(self.gy)))
        end = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
        while end == start:
            end = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
        return bfs_path(start, end, self.blocked_cells)

    def move_along_path(self):
        if self.path_idx < len(self.path):
            gx, gy = self.path[self.path_idx]
            wx = self.gx * BLOCK_SIZE
            wy = self.gy * BLOCK_SIZE
            tx = gx * BLOCK_SIZE
            ty = gy * BLOCK_SIZE
            dx = tx - wx
            dy = ty - wy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.speed:
                self.gx, self.gy = gx, gy
                self.path_idx += 1
            else:
                self.gx += dx / dist * self.speed / BLOCK_SIZE
                self.gy += dy / dist * self.speed / BLOCK_SIZE
        else:
            self.path = self.select_random_path()
            self.path_idx = 0

    def is_in_building(self):
        cell = (int(round(self.gx)), int(round(self.gy)))
        return cell == BUILDING_INSIDE_BLOCK

    def clear_contract(self):
        if self.link_target and self.link_target.startswith("E"):
            idx = int(self.link_target[1:]) - 1
            if 0 <= idx < len(self.env.edge_qs):
                self.env.edge_qs[idx] -= self.req_q
                if self.env.edge_qs[idx]<0:
                    self.env.edge_qs[idx]=0
        if self.in_cloud_contract:
            if self in self.env.cloud_contracts:
                self.env.cloud_contracts.remove(self)
            self.env.cloud_q = max(0, self.env.cloud_q - self.req_q)
        
        self.in_contract = False
        self.in_cloud_contract = False
        self.link_target = None
        self.reset_link_info()

        # 기술 플래그 초기화
        self.apply_early_exit = False
        self.apply_federated = False
        self.apply_split = False
        self.apply_personalization = False

    def reset_link_info(self):
        self.link_fading_type = None
        self.link_comm_type = None
        self.link_upload_speed = 0.0
        self.link_download_speed = 0.0
        self.fading_update_counter = 0
        self.link_path_loss_db = 0.0
        self.link_shadowing_db = 0.0
        self.rayleigh_factor = 1.0

    def attempt_contract_renewal(self):
        if not self.in_contract or self.in_cloud_contract:
            return
        
        idx = self.get_edge_idx()
        if idx is None:
            return

        up = self.env.edge_unit_prices[idx]
        current_q = self.env.edge_qs[idx]
        speed = (1 - current_q / EDGE_SERVER_MAX_Q) * 60.0
        edge_utility = self.calculate_utility("EDGE", up, speed)
        
        if edge_utility >= self.utility_threshold * (0.7 if self.loyalty > 0.5 else 0.8):
            self.remaining_time += self.req_t * (self.consistency_factor - 1.0)
            return
        
        local_utility = self.calculate_utility("LOCAL",0,0)
        cloud_load_factor = (1 - self.env.cloud_q/CLOUD_MAX_Q)
        cloud_speed_est = 50.0 * max(0.1, cloud_load_factor)
        cloud_utility = self.calculate_utility("CLOUD", p_cloud, cloud_speed_est)
        
        utilities = [
            ("EDGE", edge_utility),
            ("LOCAL", local_utility),
            ("CLOUD", cloud_utility)
        ]
        
        utilities.sort(key=lambda x: x[1], reverse=True)
        best_option, best_utility = utilities[0]
        
        if best_option != "EDGE" and best_utility > edge_utility * 1.2:
            self.clear_contract()
            if best_option == "CLOUD":
                self.try_rent_cloud()
            elif best_option == "LOCAL":
                self.state = "Idle"
            else:
                self.state = "Fail"

    def get_edge_idx(self):
        if self.link_target and self.link_target.startswith("E"):
            return int(self.link_target[1:]) - 1
        return None

    def try_rent_edge(self):
        if self.is_in_building():
            self.state = "Fail"
            return
        
        Q = sum(self.env.edge_qs)
        sx, sy = to_iso(self.gx * BLOCK_SIZE, self.gy * BLOCK_SIZE)
        order = self.env.get_edge_servers_sorted_by_distance(sx, sy)
        success = False
        
        for idx in order:
            if idx >= len(self.env.edge_unit_prices):
                continue
            
            up = self.env.edge_unit_prices[idx]
            current_q = self.env.edge_qs[idx]
            
            speed = (1 - current_q / EDGE_SERVER_MAX_Q) * 60.0
            total_cost = up * self.req_t
            edge_utility = self.calculate_utility("EDGE", up, speed)
            
            if (up > p_cloud * 2.0 or
                total_cost > self.budget or
                current_q > EDGE_SERVER_MAX_Q * 0.9):
                continue
            
            if edge_utility >= self.utility_threshold and current_q + self.req_q <= EDGE_SERVER_MAX_Q:
                self.in_contract = True
                self.in_cloud_contract = False
                self.env.edge_qs[idx] += self.req_q
                self.env.total_revenue += total_cost
                if idx < len(self.env.edge_revenues):
                    self.env.edge_revenues[idx] += total_cost
                self.env.budget -= total_cost
                self.current_price = up
                self.link_target = f"E{idx + 1}"
                self.env.log_event(f"Edge#{idx + 1} contract cost={to_musd(total_cost):.0f}m$")
                success = True
                self.setup_link("EDGE")
                if self.loyalty > 0.7:
                    self.remaining_time += self.req_t * (self.consistency_factor - 1.0)
                break
        if not success:
            self.try_rent_cloud_or_local()

    def try_rent_cloud(self):
        if self.is_in_building():
            self.state = "Fail"
            return

        if self.env.cloud_q + self.req_q > CLOUD_MAX_Q:
            self.state = "Fail"
            return

        cloud_load_factor = (1 - self.env.cloud_q/CLOUD_MAX_Q)
        cloud_speed_est = 50.0 * max(0.1, cloud_load_factor)
        final_cost = p_cloud * self.req_t
        cloud_utility = self.calculate_utility("CLOUD", p_cloud, cloud_speed_est)
        
        if cloud_utility >= self.utility_threshold and final_cost <= self.budget:
            self.in_contract = True
            self.in_cloud_contract = True
            self.env.total_revenue += final_cost
            self.env.cloud_revenue += final_cost
            self.env.budget -= final_cost
            self.current_price = p_cloud
            self.link_target = "CLOUD"
            self.env.cloud_contracts.append(self)
            self.env.cloud_q += self.req_q
            self.env.log_event(f"CLOUD contract cost={to_musd(final_cost):.0f}m$")
            self.setup_link("CLOUD")
            if self.loyalty > 0.5:
                self.remaining_time += self.req_t * (self.consistency_factor - 1.0)
        else:
            self.try_rent_cloud_or_local(fail_cloud=True)

    def try_rent_cloud_or_local(self, fail_cloud=False):
        local_utility = self.calculate_utility("LOCAL",0,0)
        cloud_load_factor = (1 - self.env.cloud_q/CLOUD_MAX_Q)
        cloud_speed_est = 50.0 * max(0.1, cloud_load_factor)
        cloud_utility = self.calculate_utility("CLOUD", p_cloud, cloud_speed_est)

        for pref in self.preference_list:
            if pref == "LOCAL":
                if local_utility >= self.utility_threshold:
                    self.state = "Idle"
                    return
            elif pref == "CLOUD" and not fail_cloud:
                if cloud_utility >= self.utility_threshold and (self.env.cloud_q+self.req_q)<=CLOUD_MAX_Q:
                    self.try_rent_cloud()
                    if self.in_cloud_contract:
                        return
            elif pref == "EDGE":
                pass
        
        self.state = "Fail"

    def setup_link(self, link_type):
        self.link_comm_type = random.choice(COMM_TECH)
        self.link_fading_type = "Rayleigh"
        if link_type == "EDGE":
            base_speed = 60.0
        else:
            base_speed = 50.0
        self.update_link_speed(base_speed)

        # Split Computing: Edge나 Cloud 사용 시 split 적용
        if link_type in ["EDGE", "CLOUD"]:
            self.apply_split = True

        # Federated Learning: Cloud + loyalty>0.5 일 경우
        if link_type == "CLOUD" and self.loyalty > 0.5:
            self.apply_federated = True

    def update_link_speed(self, base_speed):
        self.rayleigh_factor = np.random.rayleigh(RAYLEIGH_SCALE)
        dist = self.get_link_distance_m()
        self.link_shadowing_db = np.random.normal(0, SHADOWING_STD)
        if dist < D0:
            dist = D0
        self.link_path_loss_db = PL0 + 10 * N_EXP * math.log10(dist / D0) + self.link_shadowing_db
        pl_linear = 10 ** (-self.link_path_loss_db / 20.0)
        final_speed = base_speed * self.rayleigh_factor * pl_linear
        final_speed = max(0.1, final_speed)
        self.link_download_speed = final_speed
        self.link_upload_speed = final_speed / 2.0

    def get_link_distance_m(self):
        if self.link_target and self.link_target.startswith("E"):
            idx = int(self.link_target[1:]) - 1
            gx, gy, c, l, op = self.env.edge_servers_positions[idx]
            wx = gx * BLOCK_SIZE
            wy = gy * BLOCK_SIZE
        elif self.in_cloud_contract:
            sx, sy = to_iso(self.gx * BLOCK_SIZE, self.gy * BLOCK_SIZE)
            bidx = self.env.get_nearest_basestation(sx, sy)
            if bidx is not None:
                bx, by = self.env.basestations[bidx]
                wx = bx * BLOCK_SIZE
                wy = by * BLOCK_SIZE
            else:
                wx = self.gx * BLOCK_SIZE
                wy = self.gy * BLOCK_SIZE
        else:
            wx = self.gx * BLOCK_SIZE
            wy = self.gy * BLOCK_SIZE

        ux = self.gx * BLOCK_SIZE
        uy = self.gy * BLOCK_SIZE
        dist_world = math.sqrt((wx - ux)**2 + (wy - uy)**2)
        return dist_world

    def calculate_utility(self, option, unit_price, speed_estimate):
        baseline_speed = 60.0
        speed_ratio = max(0.1, speed_estimate / baseline_speed)
        
        if option=="LOCAL":
            process_time = self.req_t*2
            quality = self.base_quality*0.8
            cost = 0
            comm_latency = 0.5
            if self.loyalty>0.5:
                self.apply_personalization = True
                process_time *= 0.9
        elif option=="EDGE":
            process_time = self.req_t*(1.0/(speed_ratio))
            quality = self.base_quality*1.2
            cost = unit_price*self.req_t
            comm_latency = 0.2
            if self.apply_split or (not self.in_contract and option=="EDGE"):
                process_time *= 0.9
        else:
            process_time = self.req_t*(1.2/(speed_ratio))
            quality = self.base_quality*1.1
            cost = unit_price*self.req_t
            comm_latency = 0.3
            if self.apply_split or (not self.in_contract and option=="CLOUD"):
                process_time *= 0.9
            if self.loyalty > 0.5:
                process_time *= 0.8
                self.apply_federated = True

        latency = process_time + comm_latency
        cost_factor = math.exp(-self.beta * cost * 3.0) if cost > 0 else 1.0
        latency_factor = math.exp(-self.alpha * latency * 0.5)
        
        U = self.base_utility * quality * cost_factor * latency_factor
        return max(0, U)

    def check_and_connect_best_option(self):
        cloud_load_factor = (1 - self.env.cloud_q/CLOUD_MAX_Q)
        cloud_speed_est = 50.0 * max(0.1, cloud_load_factor)
        candidates = {
            "LOCAL": self.calculate_utility("LOCAL",0,0),
            "EDGE": self.calculate_utility("EDGE", sum(self.env.edge_unit_prices)/len(self.env.edge_unit_prices) if self.env.edge_unit_prices else 0.01, 60.0),
            "CLOUD": self.calculate_utility("CLOUD", p_cloud, cloud_speed_est)
        }

        for pref in self.preference_list:
            ut = candidates[pref]
            if ut >= self.utility_threshold:
                if pref == "LOCAL":
                    self.state = "Idle"
                    return
                elif pref == "EDGE":
                    self.try_rent_edge()
                    if self.in_contract and not self.in_cloud_contract:
                        return
                else:
                    self.try_rent_cloud()
                    if self.in_cloud_contract:
                        return
                    if self.in_contract and not self.in_cloud_contract:
                        return

        self.state = "Fail"

    def update(self):
        if not self.alive:
            return
        if PAUSED:
            return

        decay_rate = np.random.exponential(0.5)
        self.lifetime -= decay_rate * SIMULATION_SPEED

        budget_decay = max(0.01, np.random.normal(0.05, 0.02))
        self.budget -= budget_decay * SIMULATION_SPEED

        if self.contract_end_time and self.env.simulation_minutes >= self.contract_end_time:
            self.clear_contract()
            self.contract_end_time = None

        self.move_along_path()

        if self.state == "Fail":
            if self.in_contract or self.in_cloud_contract:
                self.clear_contract()
            self.alive = False
            self.kill()
            self.env.spawn_one_user()
            return

        if not (self.in_contract or self.in_cloud_contract) and self.state == "Idle":
            if random.random() < 0.01:
                self.check_and_connect_best_option()

        if self.in_contract and not self.in_cloud_contract:
            if self.remaining_time < self.req_t/2 and not self.apply_early_exit:
                if random.random()<0.3:
                    self.remaining_time *= 0.5
                    self.apply_early_exit = True

            self.remaining_time -= 1 * SIMULATION_SPEED
            if self.remaining_time % 100 == 0:
                self.attempt_contract_renewal()
            if self.remaining_time <= 0:
                if self.loyalty > 0.7:
                    idx = self.get_edge_idx()
                    if idx is not None:
                        up = self.env.edge_unit_prices[idx]
                        speed = (1 - self.env.edge_qs[idx] / EDGE_SERVER_MAX_Q)*60.0
                        edge_utility = self.calculate_utility("EDGE", up, speed)
                        if edge_utility >= self.utility_threshold:
                            self.remaining_time = self.req_t * self.consistency_factor
                            self.env.log_event("Edge contract renewed by loyalty")
                            return
                self.clear_contract()
                self.env.log_event("Edge contract ended")
                self.alive = False
                self.kill()
                self.env.spawn_one_user()
                return

        if self.in_cloud_contract:
            self.remaining_time -= 1 * SIMULATION_SPEED
            if self.remaining_time <= 0:
                if self.loyalty > 0.5:
                    cloud_load_factor = (1 - self.env.cloud_q/CLOUD_MAX_Q)
                    cloud_speed_est = 50.0 * max(0.1, cloud_load_factor)
                    cloud_utility = self.calculate_utility("CLOUD", p_cloud, cloud_speed_est)
                    if cloud_utility >= self.utility_threshold and (self.env.cloud_q+self.req_q)<=CLOUD_MAX_Q:
                        self.remaining_time = self.req_t * self.consistency_factor
                        self.env.log_event("Cloud contract renewed by loyalty")
                        return
                self.clear_contract()
                self.env.log_event("Cloud contract ended")
                self.alive = False
                self.kill()
                self.env.spawn_one_user()
                return

        if self.in_contract or self.in_cloud_contract:
            self.fading_update_counter += 1 * SIMULATION_SPEED
            if self.fading_update_counter >= FADING_UPDATE_INTERVAL:
                if self.link_target and self.link_target.startswith("E"):
                    base_speed = 60.0
                else:
                    base_speed = 50.0
                self.update_link_speed(base_speed)
                self.fading_update_counter = 0

    def draw_self(self, screen):
        ix, iy = to_iso(self.gx * BLOCK_SIZE, self.gy * BLOCK_SIZE)
        self.rect.center = (ix, iy)
        screen.blit(self.image, self.rect.topleft)

        # 머리 위에 적용 기술 상태 표시
        tech_flags = ""
        if self.apply_early_exit:
            tech_flags += "E"
        if self.apply_federated:
            tech_flags += "F"
        if self.apply_split:
            tech_flags += "S"
        if self.apply_personalization:
            tech_flags += "P"

        if tech_flags:
            font_sm = pygame.font.SysFont(None, 20)
            txt_surf = font_sm.render(tech_flags, True, (0,0,0))
            screen.blit(txt_surf, (ix - txt_surf.get_width()//2, iy - 20))


class BuildingOccupant:
    def __init__(self, env):
        self.env = env
        self.task_type = random.choice(["AR","VR","ML","Voice"])
        
        base_req = random.randint(100, 300)
        self.req_q = int(base_req * TASK_GPU_WEIGHTS[self.task_type])
        
        self.req_gpu = self.req_q * GPU_PER_UNIT
        self.req_vram = self.req_q * VRAM_PER_UNIT
        self.req_cpu = self.req_q * CPU_PER_UNIT
        self.req_ram = self.req_q * RAM_PER_UNIT
        self.req_storage = self.req_q * STORAGE_PER_UNIT
        
        self.req_t = random.randint(30, 180)
        self.remaining_time = self.req_t
        self.elapsed_time = 0
        
        self.in_contract = False
        self.in_cloud_contract = False
        self.current_price = 0
        self.link_target = None
        self.alive = True
        self.state = "Idle"

    def update(self):
        if PAUSED:
            return
        occupant_count = self.env.building_occupants_count()
        speed_factor = max(0.1, 1.0/(1+(occupant_count/10)))
        self.link_download_speed = 200.0 * speed_factor
        self.link_upload_speed = 100.0 * speed_factor

        self.remaining_time -= SIMULATION_SPEED
        if self.remaining_time <= 0:
            self.leave_building()

    def leave_building(self):
        self.alive = False
        self.env.building_occupants_list.remove(self)
        gx, gy = BUILDING_INSIDE_BLOCK
        gx_out = gx + random.randint(-1,1)
        gy_out = gy + random.randint(-1,1)
        c = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        p = Person(gx_out, gy_out, c, self.env, self.env.road_nodes, self.env.blocked_cells, from_building=True)
        self.env.people.add(p)


class MECEnvironment:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 32)
        self.people = pygame.sprite.Group()
        self.cloud_contracts = []
        self.edge_qs = [0, 0, 0]
        self.current_N = TOTAL_CORES
        self.total_revenue = 0.0
        self.edge_revenues = [0.0, 0.0, 0.0]
        self.cloud_revenue = 0.0
        self.budget = 0.0
        
        self.last_edge_qs = [0]*3
        self.last_edge_prices = [0.01, 0.01, 0.01]
        
        self.simulation_minutes = 0

        self.opt_q, self.opt_p, self.opt_prof = find_optimal_q_for_N(self.current_N)
        self.events = collections.deque(maxlen=50)

        # 로그 파일 초기화
        self.log_file = open("log.txt", "a", encoding="utf-8")
        self.log_event("Simulation started")

        self.road_nodes = []
        for gx in range(GRID_W):
            for gy in range(GRID_H):
                self.road_nodes.append((gx, gy))

        self.decorations = []
        building_positions = []
        for i in range(BUILDING_COUNT):
            bx = random.randint(1, GRID_W - 2)
            by = random.randint(1, GRID_H - 2)
            bw = random.randint(60, 80)
            bh = random.randint(100, 150)
            self.decorations.append(('building', (bx, by, bw, bh)))
            building_positions.append((bx, by, bw, bh))

        for i in range(TREE_COUNT):
            tx = random.randint(1, GRID_W - 2)
            ty = random.randint(1, GRID_H - 2)
            r = random.randint(10, 20)
            self.decorations.append(('tree', (tx, ty, r)))

        for i in range(NOISE_POINTS):
            nx = random.uniform(0, GRID_W)
            ny = random.uniform(0, GRID_H)
            cc = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            self.decorations.append(('noise', (nx, ny, cc)))

        self.basestations = []
        for i in range(BASESTATION_COUNT):
            while True:
                bx = random.randint(0, GRID_W - 1)
                by = random.randint(0, GRID_H - 1)
                if (bx, by) not in self.basestations:
                    self.basestations.append((bx, by))
                    self.decorations.append(('basestation', (bx, by)))
                    break

        chosen_buildings = random.sample(building_positions, EDGE_SERVERS_COUNT)
        self.edge_servers_positions = []
        for i, (bx, by, bw, bh) in enumerate(chosen_buildings):
            c_color = [(200, 50, 50), (50, 200, 50), (50, 50, 200)][i % 3]
            lbl = f"E{i + 1}"
            operator_name = OPERATORS[i % len(OPERATORS)]
            self.edge_servers_positions.append((bx, by, c_color, lbl, operator_name))

        self.blocked_cells = set((p[0], p[1]) for p in self.edge_servers_positions)

        for i in range(MAX_INITIAL_PEOPLE):
            self.spawn_one_user()

        self.cloud_server_box = pygame.Rect(MAIN_AREA_WIDTH - 300, 50, 200, 150)
        self.log_panel_rect = pygame.Rect(MAIN_AREA_WIDTH, SCREEN_HEIGHT - 300, RIGHT_PANEL_WIDTH, 300)

        self.building_set = set((b[0], b[1]) for b in building_positions)

        self.edge_unit_prices = [0.01, 0.01, 0.01]
        self.recalculate_edge_prices()

        self.menu = None
        self.adding_edge_server = False
        self.adding_basestation = False

        # ------------------------------ #
        # 데이터 수집용 구조체 확장
        # ------------------------------ #
        self.data = {
            'time': [],
            'total_revenue': [],
            'cloud_revenue': [],
            'edge_revenues': [[] for _ in range(EDGE_SERVERS_COUNT)],
            'edge_loads': [[] for _ in range(EDGE_SERVERS_COUNT)],
            'user_count': [],
            'cloud_users': [],
            'edge_users': [],
            'local_users': [],
            # 평균 링크 속도 추적
            'avg_edge_down_speed': [],
            'avg_edge_up_speed': [],
            'avg_cloud_down_speed': [],
            'avg_cloud_up_speed': [],
            # 태스크 유형 분포
            'count_AR': [],
            'count_VR': [],
            'count_ML': [],
            'count_Voice': [],
        }
        # 데이터 수집 주기
        self.data_collection_interval = 60
        self.last_collection_time = 0

        self.speed_buttons = [
            ("1x", (MAIN_AREA_WIDTH - 200, 200), 1),
            ("2x", (MAIN_AREA_WIDTH - 130, 200), 2),
            ("4x", (MAIN_AREA_WIDTH - 60, 200), 4),
            ("8x",(MAIN_AREA_WIDTH - 200,240),8),
            ("16x",(MAIN_AREA_WIDTH -130,240),16),
            ("32x",(MAIN_AREA_WIDTH - 60,240),32),
            ("64x",(MAIN_AREA_WIDTH -200,280),64),
            ("128x",(MAIN_AREA_WIDTH -130,280),128),
            ("Pause",(MAIN_AREA_WIDTH -60,280),None)
        ]

        self.link_lines = []
        self.building_occupants_list = []
        self.building_arrival_rate = 5.0
        self.cloud_q = 0

    def building_occupants_count(self):
        return len(self.building_occupants_list)

    def spawn_one_user(self):
        x = random.randint(0, GRID_W - 1)
        y = random.randint(0, GRID_H - 1)
        if (x, y) in self.blocked_cells:
            return self.spawn_one_user()
        c = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        p = Person(x, y, c, self, self.road_nodes, self.blocked_cells)
        self.people.add(p)

    def spawn_building_occupant(self):
        o = BuildingOccupant(self)
        self.building_occupants_list.append(o)

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def get_minutes(self):
        return self.simulation_minutes

    def get_day_time_str(self):
        m = self.get_minutes()
        total_minutes = int(m) + START_MINUTE + (START_HOUR * 60)
        days = total_minutes // (24 * 60)
        remain = total_minutes % (24 * 60)
        hh = remain // 60
        mm = remain % 60
        ap = "AM"
        if hh >= 12:
            ap = "PM"
        h12 = (hh % 12)
        if h12 == 0:
            h12 = 12
        return f"{days} days {h12:02d}:{mm:02d}{ap}"

    def get_hour(self):
        m = self.get_minutes()
        total_minutes = int(m) + START_MINUTE + (START_HOUR * 60)
        hh = total_minutes // 60
        return hh % 24

    def log_event(self, msg):
        if "contract" in msg or "ended" in msg or "CLOUD" in msg or "Edge" in msg:
            self.events.append(f"{int(self.get_minutes())}m: {msg}")
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.write(f"{int(self.get_minutes())}m: {msg}\n")
                self.log_file.flush()

    def get_edge_servers_sorted_by_distance(self, sx, sy):
        servers_iso = []
        for i, (gx, gy, c, lbl, op) in enumerate(self.edge_servers_positions):
            wx = gx * BLOCK_SIZE
            wy = gy * BLOCK_SIZE
            ix, iy = to_iso(wx, wy)
            dist = (ix - sx) ** 2 + (iy - sy) ** 2
            servers_iso.append((dist, i))
        servers_iso.sort(key=lambda x: x[0])
        return [x[1] for x in servers_iso]

    def get_nearest_basestation(self, sx, sy):
        best_dist = 1e9
        best_idx = None
        for i, (bx, by) in enumerate(self.basestations):
            wx = bx * BLOCK_SIZE
            wy = by * BLOCK_SIZE
            ix, iy = to_iso(wx, wy)
            dist = (ix - sx) ** 2 + (iy - sy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def recalculate_edge_prices(self):
        Q_total = sum(self.edge_qs)
        N = self.current_N
        
        for i, (gx, gy, c, lbl, op) in enumerate(self.edge_servers_positions):
            old_price = self.edge_unit_prices[i]
            current_q = self.edge_qs[i]
            
            total_gpu_usage = current_q * GPU_PER_UNIT
            power_usage = 0.3 + 0.7 * (total_gpu_usage / (EDGE_SERVER_MAX_Q * GPU_PER_UNIT))
            power_cost = power_usage * POWER_COST_PER_HOUR
            
            marginal_cost = (power_cost / max(1, current_q)) * (
                GPU_PER_UNIT * 10 +
                VRAM_PER_UNIT/1024/5 +
                CPU_PER_UNIT +
                RAM_PER_UNIT/1024/10
            )
            
            current_revenue_per_hour = current_q * old_price * 60
            current_profit_per_hour = current_revenue_per_hour - power_cost
            
            if self.last_edge_prices[i] != old_price and self.last_edge_prices[i] > 0:
                price_change = (old_price - self.last_edge_prices[i]) / self.last_edge_prices[i]
                if price_change != 0:
                    demand_change = (current_q - self.last_edge_qs[i]) / max(1, self.last_edge_qs[i])
                    elasticity = demand_change / price_change
                else:
                    elasticity = -1.0
            else:
                elasticity = -1.0
                
            if elasticity != 0:
                optimal_markup = abs(1/elasticity)
                new_price = marginal_cost * (1 + optimal_markup)
            else:
                new_price = old_price
                
            min_price = p_cloud * 0.1
            max_price = p_cloud * 2.0
            new_price = max(min_price, min(new_price, max_price))
            
            max_change = 0.15
            if new_price > old_price:
                final_price = min(new_price, old_price * (1 + max_change))
            else:
                final_price = max(new_price, old_price * (1 - max_change))
                
            self.last_edge_qs[i] = current_q
            self.last_edge_prices[i] = old_price
            self.edge_unit_prices[i] = final_price

    def step(self):
        if PAUSED:
            return
        for _ in range(SIMULATION_SPEED):
            self.simulation_minutes += 1
            
            hh = self.get_hour()
            if hh >= NIGHT_START or hh < NIGHT_END:
                for p in list(self.people):
                    if random.random() < 0.0005 * np.random.exponential(1.0):
                        if p.in_contract or p.in_cloud_contract:
                            p.clear_contract()
                        p.alive = False
                        p.kill()
            else:
                arr_rate = get_arrival_rate(hh)
                arrivals = np.random.poisson(arr_rate/60)
                
                for _ in range(arrivals):
                    if random.random() < np.random.exponential(1.0)/2:
                        self.spawn_one_user()

            b_arrivals = np.random.poisson(self.building_arrival_rate/60)
            for _ in range(b_arrivals):
                if random.random() < np.random.exponential(1.0)/2:
                    self.spawn_building_occupant()

            for p in list(self.people):
                p.update()

            for o in list(self.building_occupants_list):
                o.update()

            self.cloud_contracts = [c for c in self.cloud_contracts if c.alive]

            if self.simulation_minutes % PRICE_CHECK_INTERVAL == 0:
                self.recalculate_edge_prices()

            self.collect_data()

    def collect_data(self):
        """1분마다(기본 60시뮬레이션분 단위) 주요 데이터를 수집하여 그래프를 그릴 수 있도록 저장."""
        if self.simulation_minutes - self.last_collection_time >= self.data_collection_interval:
            self.data['time'].append(self.simulation_minutes)
            self.data['total_revenue'].append(self.total_revenue)
            self.data['cloud_revenue'].append(self.cloud_revenue)

            for i in range(len(self.edge_revenues)):
                if i < len(self.data['edge_revenues']):
                    self.data['edge_revenues'][i].append(self.edge_revenues[i])
                    self.data['edge_loads'][i].append(self.edge_qs[i])

            alive_people = [p for p in self.people if p.alive]

            # 사용자 수
            alive_count = len(alive_people)
            cloud_users = len(self.cloud_contracts)
            edge_users = sum(p.in_contract and not p.in_cloud_contract for p in alive_people)
            local_users = sum((not p.in_contract) and (not p.in_cloud_contract) and (p.state=="Idle") for p in alive_people)

            self.data['user_count'].append(alive_count)
            self.data['cloud_users'].append(cloud_users)
            self.data['edge_users'].append(edge_users)
            self.data['local_users'].append(local_users)

            # 평균 엣지/클라우드 속도
            edge_down_list = []
            edge_up_list = []
            cloud_down_list = []
            cloud_up_list = []

            for p in alive_people:
                if p.in_contract and not p.in_cloud_contract:
                    edge_down_list.append(p.link_download_speed)
                    edge_up_list.append(p.link_upload_speed)
                elif p.in_cloud_contract:
                    cloud_down_list.append(p.link_download_speed)
                    cloud_up_list.append(p.link_upload_speed)

            avg_edge_down = np.mean(edge_down_list) if len(edge_down_list)>0 else 0
            avg_edge_up = np.mean(edge_up_list) if len(edge_up_list)>0 else 0
            avg_cloud_down = np.mean(cloud_down_list) if len(cloud_down_list)>0 else 0
            avg_cloud_up = np.mean(cloud_up_list) if len(cloud_up_list)>0 else 0

            self.data['avg_edge_down_speed'].append(avg_edge_down)
            self.data['avg_edge_up_speed'].append(avg_edge_up)
            self.data['avg_cloud_down_speed'].append(avg_cloud_down)
            self.data['avg_cloud_up_speed'].append(avg_cloud_up)

            # 태스크 유형 분포
            cnt_AR = sum(1 for p in alive_people if p.task_type == "AR")
            cnt_VR = sum(1 for p in alive_people if p.task_type == "VR")
            cnt_ML = sum(1 for p in alive_people if p.task_type == "ML")
            cnt_Voice = sum(1 for p in alive_people if p.task_type == "Voice")

            self.data['count_AR'].append(cnt_AR)
            self.data['count_VR'].append(cnt_VR)
            self.data['count_ML'].append(cnt_ML)
            self.data['count_Voice'].append(cnt_Voice)

            self.last_collection_time = self.simulation_minutes

    def save_and_plot_data(self):
        """시뮬레이션 결과를 CSV, PNG로 저장하고, 1920x1080 해상도 정도로 그래프를 한 페이지에 표시한 뒤 plt.show() 실행."""
        if not os.path.exists("results"):
            os.makedirs("results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        import pandas as pd
        df = pd.DataFrame({
            'time': self.data['time'],
            'total_revenue': self.data['total_revenue'],
            'cloud_revenue': self.data['cloud_revenue'],
            'user_count': self.data['user_count'],
            'cloud_users': self.data['cloud_users'],
            'edge_users': self.data['edge_users'],
            'local_users': self.data['local_users'],
            'avg_edge_down_speed': self.data['avg_edge_down_speed'],
            'avg_edge_up_speed': self.data['avg_edge_up_speed'],
            'avg_cloud_down_speed': self.data['avg_cloud_down_speed'],
            'avg_cloud_up_speed': self.data['avg_cloud_up_speed'],
            'count_AR': self.data['count_AR'],
            'count_VR': self.data['count_VR'],
            'count_ML': self.data['count_ML'],
            'count_Voice': self.data['count_Voice'],
        })
        for i in range(len(self.edge_revenues)):
            df[f'edge_{i+1}_revenue'] = self.data['edge_revenues'][i]
            df[f'edge_{i+1}_load'] = self.data['edge_loads'][i]
        
        csv_path = f"results/simulation_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # 1920x1080 그래프
        plt.figure(figsize=(19.2, 10.8), dpi=100)  # 대략 1920x1080 크기에 해당

        # 3x3 레이아웃
        plt.subplot(3,3,1)
        plt.plot(self.data['time'], self.data['total_revenue'], label='Total Revenue')
        plt.plot(self.data['time'], self.data['cloud_revenue'], label='Cloud Revenue')
        for i in range(len(self.edge_revenues)):
            plt.plot(self.data['time'], self.data['edge_revenues'][i], label=f'Edge {i+1} Revenue')
        plt.xlabel('Time (min)')
        plt.ylabel('Revenue ($)')
        plt.title('Revenue Over Time')
        plt.legend()

        plt.subplot(3,3,2)
        for i in range(len(self.edge_qs)):
            plt.plot(self.data['time'], self.data['edge_loads'][i], label=f'Edge {i+1} Load')
        plt.xlabel('Time (min)')
        plt.ylabel('Queue Size')
        plt.title('Edge Server Loads')
        plt.legend()

        plt.subplot(3,3,3)
        plt.plot(self.data['time'], self.data['user_count'], label='Total Users')
        plt.plot(self.data['time'], self.data['cloud_users'], label='Cloud Users')
        plt.plot(self.data['time'], self.data['edge_users'], label='Edge Users')
        plt.plot(self.data['time'], self.data['local_users'], label='Local Only')
        plt.xlabel('Time (min)')
        plt.ylabel('Users')
        plt.title('User Count')
        plt.legend()

        plt.subplot(3,3,4)
        plt.plot(self.data['time'], self.data['avg_edge_down_speed'], label='Edge Down')
        plt.plot(self.data['time'], self.data['avg_edge_up_speed'], label='Edge Up')
        plt.xlabel('Time (min)')
        plt.ylabel('Mbps')
        plt.title('Avg Edge Speeds')
        plt.legend()

        plt.subplot(3,3,5)
        plt.plot(self.data['time'], self.data['avg_cloud_down_speed'], label='Cloud Down')
        plt.plot(self.data['time'], self.data['avg_cloud_up_speed'], label='Cloud Up')
        plt.xlabel('Time (min)')
        plt.ylabel('Mbps')
        plt.title('Avg Cloud Speeds')
        plt.legend()

        plt.subplot(3,3,6)
        plt.plot(self.data['time'], self.data['count_AR'], label='AR')
        plt.plot(self.data['time'], self.data['count_VR'], label='VR')
        plt.plot(self.data['time'], self.data['count_ML'], label='ML')
        plt.plot(self.data['time'], self.data['count_Voice'], label='Voice')
        plt.xlabel('Time (min)')
        plt.ylabel('Active Users')
        plt.title('Task Type Distribution')
        plt.legend()

        plt.subplot(3,3,7)
        for i in range(len(self.edge_revenues)):
            plt.plot(self.data['time'], self.data['edge_revenues'][i], label=f'Edge {i+1}')
        plt.xlabel('Time (min)')
        plt.ylabel('Revenue ($)')
        plt.title('Edge Revenues Comparison')
        plt.legend()

        plt.subplot(3,3,8)
        cloud_usage = [cq for cq in df['cloud_users']]
        plt.plot(self.data['time'], cloud_usage, label='Cloud Users')
        plt.xlabel('Time (min)')
        plt.ylabel('Users in Cloud')
        plt.title('Cloud Usage Over Time')
        plt.legend()

        plt.subplot(3,3,9)
        plt.plot(self.data['time'], self.data['total_revenue'], label='Total Revenue', color='red')
        plt.xlabel('Time (min)')
        plt.ylabel('Revenue ($)')
        plt.title('Total Revenue (Zoomed)')
        plt.legend()

        plt.tight_layout()

        png_path = f"results/simulation_results_{timestamp}.png"
        plt.savefig(png_path)

        plt.show()

    def draw_isometric_building(self, bx, by, bw, bh):
        wx = (bx + 0.5) * BLOCK_SIZE
        wy = (by + 0.5) * BLOCK_SIZE
        ix, iy = to_iso(wx, wy)
        top_w = bw
        top_h = bw // 2
        top_poly = [
            (ix - top_w // 2, iy - bh - top_h // 2),
            (ix + top_w // 2, iy - bh - top_h // 2),
            (ix + top_w // 2, iy - bh + top_h // 2),
            (ix - top_w // 2, iy - bh + top_h // 2)
        ]
        top_color = (180, 180, 220)
        pygame.draw.polygon(self.screen, top_color, top_poly)

        left_poly = [
            (ix - top_w // 2, iy - bh - top_h // 2),
            (ix - top_w // 2, iy - bh + top_h // 2),
            (ix - top_w // 2 + 10, iy + top_h // 2),
            (ix - top_w // 2 + 10, iy - top_h // 2)
        ]
        left_color = (130, 130, 180)
        pygame.draw.polygon(self.screen, left_color, left_poly)

        right_poly = [
            (ix + top_w // 2, iy - bh - top_h // 2),
            (ix + top_w // 2, iy - bh + top_h // 2),
            (ix + top_w // 2 - 10, iy + top_h // 2),
            (ix + top_w // 2 - 10, iy - top_h // 2)
        ]
        right_color = (120, 120, 170)
        pygame.draw.polygon(self.screen, right_color, right_poly)

        front_poly = [
            (ix - top_w // 2 + 10, iy + top_h // 2),
            (ix + top_w // 2 - 10, iy + top_h // 2),
            (ix + top_w // 2 - 10, iy + top_h // 2 + bh // 2),
            (ix - top_w // 2 + 10, iy + top_h // 2 + bh // 2)
        ]
        front_color = (100, 100, 160)
        pygame.draw.polygon(self.screen, front_color, front_poly)

    def draw_isometric_tree(self, tx, ty, r):
        wx = (tx + 0.7) * BLOCK_SIZE
        wy = (ty + 0.7) * BLOCK_SIZE
        ix, iy = to_iso(wx, wy)
        pygame.draw.rect(self.screen, (139, 69, 19), (ix - 3, iy, 6, r // 2))
        pygame.draw.circle(self.screen, (34, 179, 34), (ix, iy), r)

    def draw_noise(self, nx, ny, cc):
        wx = nx * BLOCK_SIZE
        wy = ny * BLOCK_SIZE
        ix, iy = to_iso(wx, wy)
        pygame.draw.rect(self.screen, cc, (ix, iy, 2, 2))

    def draw_basestation(self, bx, by):
        wx = (bx + 0.5) * BLOCK_SIZE
        wy = (by + 0.5) * BLOCK_SIZE
        ix, iy = to_iso(wx, wy)
        pygame.draw.rect(self.screen, (120, 120, 120), (ix - 8, iy - 45, 16, 45))
        pygame.draw.circle(self.screen, (200, 0, 0), (ix, iy - 50), 8)

    def draw_server_octagon(self, ix, iy, color, label=None):
        pygame.draw.rect(self.screen, (180, 180, 180), (ix - 25, iy - 25, 50, 50), 0)
        offs = [(-20, -10), (-10, -20), (10, -20), (20, -10), (20, 10), (10, 20), (-10, 20), (-20, 10)]
        points = []
        for ox, oy in offs:
            points.append((ix + ox, iy + oy))
        pygame.draw.polygon(self.screen, color, points)

        if label:
            fnt_sm = pygame.font.SysFont(None, 32)
            lbl_r = fnt_sm.render(label, True, (0, 0, 0))
            self.screen.blit(lbl_r, (ix - lbl_r.get_width() // 2, iy - lbl_r.get_height() // 2))

    def get_building_intersections_world(self, x1, y1, x2, y2):
        def bresenham(x0, y0, x1, y1):
            cells = []
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x, y = x0, y0
            while True:
                cells.append((x, y))
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy
            return cells

        line_cells = bresenham(x1, y1, x2, y2)
        count = 0
        for (cx, cy) in line_cells:
            if (cx, cy) in self.building_set:
                count += 1
        return count

    def draw_contract_lines(self):
        self.link_lines.clear()
        cloud_pos = (self.cloud_server_box.x + self.cloud_server_box.width // 2,
                     self.cloud_server_box.y + self.cloud_server_box.height // 2)

        for p in self.people:
            if p.alive and (p.in_contract or p.in_cloud_contract):
                sx, sy = to_iso(p.gx * BLOCK_SIZE, p.gy * BLOCK_SIZE)
                if p.link_target:
                    if p.link_target.startswith("E"):
                        idx = int(p.link_target[1:]) - 1
                        if idx >= len(self.edge_servers_positions):
                            continue
                        gx, gy, c, l, op = self.edge_servers_positions[idx]
                        nb = self.get_building_intersections_world(int(round(p.gx)), int(round(p.gy)), gx, gy)
                        intensity = max(0, 255 - nb * 30)
                        wx = gx * BLOCK_SIZE
                        wy = gy * BLOCK_SIZE
                        ix, iy = to_iso(wx, wy)
                        if idx == 0:
                            line_color = (200, 50, 50)
                        elif idx == 1:
                            line_color = (50, 200, 50)
                        else:
                            line_color = (50, 50, 200)
                        line_color = tuple(min(255, int(c_val * intensity / 255)) for c_val in line_color)
                        pygame.draw.line(self.screen, line_color, (sx, sy), (ix, iy), 2)
                        self.link_lines.append(((sx, sy), (ix, iy), p))
                    elif p.link_target == "CLOUD":
                        bidx = self.get_nearest_basestation(sx, sy)
                        if bidx is not None and bidx < len(self.basestations):
                            bx, by = self.basestations[bidx]
                            nb = self.get_building_intersections_world(int(round(p.gx)), int(round(p.gy)), bx, by)
                            intensity = max(0, 255 - nb * 30)
                            wx = bx * BLOCK_SIZE
                            wy = by * BLOCK_SIZE
                            bix, biy = to_iso(wx, wy)
                            pygame.draw.line(self.screen, (intensity, 0, 0), (sx, sy), (bix, biy), 1)
                            pygame.draw.line(self.screen, (0, 0, 0), (bix, biy), cloud_pos, 1)
                            self.link_lines.append(((sx, sy), (bix, biy), p))
                        else:
                            pygame.draw.line(self.screen, (0, 0, 0), (sx, sy), cloud_pos, 1)
                            self.link_lines.append(((sx, sy), cloud_pos, p))

    def draw_roads(self):
        for gx in range(GRID_W):
            for gy in range(GRID_H):
                if gx < GRID_W - 1:
                    sx, sy = to_iso(gx * BLOCK_SIZE, gy * BLOCK_SIZE)
                    ex, ey = to_iso((gx + 1) * BLOCK_SIZE, gy * BLOCK_SIZE)
                    pygame.draw.line(self.screen, (80, 80, 80), (sx, sy), (ex, ey), 4)
                if gy < GRID_H - 1:
                    sx, sy = to_iso(gx * BLOCK_SIZE, gy * BLOCK_SIZE)
                    ex, ey = to_iso(gx * BLOCK_SIZE, (gy + 1) * BLOCK_SIZE)
                    pygame.draw.line(self.screen, (80, 80, 80), (sx, sy), (ex, ey), 4)

    def draw_top_panel(self):
        alive_count = sum(1 for p in self.people if p.alive)
        current_time_str = self.get_day_time_str()
        fnt_sm = pygame.font.SysFont(None, 32)
        box_color = (230, 230, 230)

        hh = self.get_hour()
        day_phase = "Day"
        if hh >= NIGHT_START or hh < NIGHT_END:
            day_phase = "Night"

        def draw_info_box(x, y, w, h, text):
            box = pygame.Surface((w, h), pygame.SRCALPHA)
            box.fill(box_color)
            tr = fnt_sm.render(text, True, (0, 0, 0))
            box.blit(tr, (10, h // 2 - tr.get_height() // 2))
            self.screen.blit(box, (x, y))

        draw_info_box(20, 20, 180, 50, f"Alive: {alive_count}")
        draw_info_box(220, 20, 300, 50, f"Time: {current_time_str}")
        draw_info_box(540, 20, 200, 50, day_phase)
        draw_info_box(760, 20, 300, 50, f"CloudRev: {to_musd(self.cloud_revenue):.0f}m$")

        pause_text = "Paused" if PAUSED else "Running"
        speed_text = f"Speed: {SIMULATION_SPEED}x"
        draw_info_box(1080, 20, 200, 50, pause_text)
        draw_info_box(1300, 20, 200, 50, speed_text)

        # 기술 적용 사용자 카운트 집계
        early_exit_count = sum(p.apply_early_exit for p in self.people if p.alive)
        federated_count = sum(p.apply_federated for p in self.people if p.alive)
        split_count = sum(p.apply_split for p in self.people if p.alive)
        pers_count = sum(p.apply_personalization for p in self.people if p.alive)

        draw_info_box(20, 80, 180, 30, f"EarlyExit: {early_exit_count}")
        draw_info_box(220, 80, 180, 30, f"Federated: {federated_count}")
        draw_info_box(420, 80, 180, 30, f"Split: {split_count}")
        draw_info_box(620, 80, 180, 30, f"Personal: {pers_count}")

        start_x = 20
        start_y = 120
        gap = 300
        Qs = self.edge_qs
        N = TOTAL_CORES
        for i, (gx, gy, c, lbl, op) in enumerate(self.edge_servers_positions):
            srv_box = pygame.Surface((250, 140), pygame.SRCALPHA)
            srv_box.fill(box_color)

            lbl_r = fnt_sm.render(f"{lbl}-{op}", True, (0, 0, 0))
            srv_box.blit(lbl_r, (10, 10))

            rev_r = fnt_sm.render(f"Rev: {to_musd(self.edge_revenues[i]):.0f}m$", True, (0, 0, 0))
            srv_box.blit(rev_r, (10, 30))

            price_r = fnt_sm.render(f"Price: {to_musd(self.edge_unit_prices[i]):.0f}m$/unit", True, (0, 0, 0))
            srv_box.blit(price_r, (10, 50))

            speed = (1 - Qs[i] / EDGE_SERVER_MAX_Q) * 60.0
            speed_r = fnt_sm.render(f"Speed: {speed:.1f}Mbps", True, (0, 0, 0))
            srv_box.blit(speed_r, (10, 70))

            bar_width = 230
            bar_height = 15
            pygame.draw.rect(srv_box, (200, 200, 200), (10, 115, bar_width, bar_height))
            fill_width = int(bar_width * (Qs[i] / EDGE_SERVER_MAX_Q))
            if fill_width > 0:
                fill_ratio = Qs[i] / EDGE_SERVER_MAX_Q
                red = min(255, int(128 + (127 * fill_ratio)))
                green = max(0, int(128 - (128 * fill_ratio)))
                pygame.draw.rect(srv_box, (red, green, 0), (10, 115, fill_width, bar_height))

            q_text = f"{Qs[i]}/{EDGE_SERVER_MAX_Q}"
            q_text_r = fnt_sm.render(q_text, True, (0, 0, 0))
            srv_box.blit(q_text_r, (10 + bar_width // 2 - q_text_r.get_width() // 2, 115 - q_text_r.get_height() - 2))

            self.screen.blit(srv_box, (start_x + i * gap, start_y))

    def draw_log_panel(self):
        log_bg = pygame.Surface((self.log_panel_rect.width, self.log_panel_rect.height), pygame.SRCALPHA)
        log_bg.fill((240, 240, 255, 200))
        self.screen.blit(log_bg, (self.log_panel_rect.x, self.log_panel_rect.y))
        fnt_sm = pygame.font.SysFont(None, 24)
        hdr = fnt_sm.render("LOG", True, (0, 0, 0))
        self.screen.blit(hdr, (self.log_panel_rect.x + 10, self.log_panel_rect.y + 10))
        line_y = self.log_panel_rect.y + 40
        logs_to_show = list(self.events)[-10:]
        for e in reversed(logs_to_show):
            r = fnt_sm.render(str(e), True, (0, 0, 0))
            self.screen.blit(r, (self.log_panel_rect.x + 10, line_y))
            line_y += r.get_height() + 2
            if line_y > self.log_panel_rect.y + self.log_panel_rect.height - 10:
                break

    def draw_cloud_box(self):
        cloud_bg = pygame.Surface((self.cloud_server_box.width, self.cloud_server_box.height), pygame.SRCALPHA)
        cloud_bg.fill((220, 220, 255, 200))
        self.screen.blit(cloud_bg, (self.cloud_server_box.x, self.cloud_server_box.y))
        fnt_sm = pygame.font.SysFont(None, 24)

        tr = fnt_sm.render("CLOUD", True, (0, 0, 0))
        self.screen.blit(tr, (self.cloud_server_box.x + self.cloud_server_box.width // 2 - tr.get_width() // 2,
                              self.cloud_server_box.y + 5))

        price_text = f"Price: {to_musd(p_cloud):.0f}m$/unit"
        price_r = fnt_sm.render(price_text, True, (0, 0, 0))
        self.screen.blit(price_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 30))

        cloud_users = len(self.cloud_contracts)
        cloud_user_txt = f"Cloud Users: {cloud_users}"
        cloud_user_r = fnt_sm.render(cloud_user_txt, True, (0, 0, 0))
        self.screen.blit(cloud_user_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 50))

        cloud_load_factor = (1 - self.cloud_q/CLOUD_MAX_Q)
        avg_down = 50.0 * max(0.1, cloud_load_factor)
        avg_up = avg_down/2.0
        cloud_speed_txt = f"Est Speed D/U: {avg_down:.1f}/{avg_up:.1f} Mbps"
        cloud_speed_r = fnt_sm.render(cloud_speed_txt, True, (0, 0, 0))
        self.screen.blit(cloud_speed_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 70))

        cloud_revenue_txt = f"Cloud Rev: {to_musd(self.cloud_revenue):.0f}m$"
        cloud_revenue_r = fnt_sm.render(cloud_revenue_txt, True, (0,0,0))
        self.screen.blit(cloud_revenue_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 90))

        cloud_capacity_txt = f"Cloud Q: {self.cloud_q}/{CLOUD_MAX_Q}"
        cloud_capacity_r = fnt_sm.render(cloud_capacity_txt, True, (0,0,0))
        self.screen.blit(cloud_capacity_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 110))

        building_wifi_users = self.building_occupants_count()
        wifi_txt = f"Bldg Wi-Fi Users: {building_wifi_users}"
        wifi_r = fnt_sm.render(wifi_txt, True, (0,0,0))
        self.screen.blit(wifi_r, (self.cloud_server_box.x + 10, self.cloud_server_box.y + 130))

    def handle_input(self):
        global camera_offset_x, camera_offset_y, ISO_SCALE
        keys = pygame.key.get_pressed()

        if keys[pygame.K_s]:
            camera_offset_x -= CAMERA_SPEED
            camera_offset_y -= CAMERA_SPEED
        if keys[pygame.K_w]:
            camera_offset_x += CAMERA_SPEED
            camera_offset_y += CAMERA_SPEED
        if keys[pygame.K_d]:
            camera_offset_x -= CAMERA_SPEED
            camera_offset_y += CAMERA_SPEED
        if keys[pygame.K_a]:
            camera_offset_x += CAMERA_SPEED
            camera_offset_y -= CAMERA_SPEED

    def apply_night_overlay(self):
        hour = self.get_hour()
        if hour >= NIGHT_START or hour < NIGHT_END:
            overlay = pygame.Surface((MAIN_AREA_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))

    def draw(self):
        self.screen.fill((255, 255, 255))

        self.draw_roads()

        for d in self.decorations:
            t = d[0]
            if t == 'building':
                bx, by, bw, bh = d[1]
                self.draw_isometric_building(bx, by, bw, bh)
            elif t == 'tree':
                tx, ty, r = d[1]
                self.draw_isometric_tree(tx, ty, r)
            elif t == 'noise':
                nx, ny, cc = d[1]
                self.draw_noise(nx, ny, cc)
            elif t == 'basestation':
                bx, by = d[1]
                self.draw_basestation(bx, by)

        for i, (gx, gy, c, lbl, op) in enumerate(self.edge_servers_positions):
            wx = gx * BLOCK_SIZE
            wy = gy * BLOCK_SIZE
            ix, iy = to_iso(wx, wy)
            self.draw_server_octagon(ix, iy, color=c, label=lbl)

        for p in self.people:
            p.draw_self(self.screen)

        self.draw_contract_lines()
        self.draw_top_panel()
        self.draw_cloud_box()
        self.draw_log_panel()
        self.draw_right_panel()
        self.draw_speed_buttons()

        if self.menu and self.menu.visible:
            self.menu.draw(self.screen)

        self.apply_night_overlay()

        if PAUSED and SHOW_CONNECTION_DETAILS:
            self.draw_connection_details()

        if PAUSED and SELECTED_LINK is not None:
            self.draw_link_popup(SELECTED_LINK)

    def draw_connection_details(self):
        detail_rect = pygame.Rect(MAIN_AREA_WIDTH, 300, RIGHT_PANEL_WIDTH, SCREEN_HEIGHT - 600)
        detail_bg = pygame.Surface((detail_rect.width, detail_rect.height), pygame.SRCALPHA)
        detail_bg.fill((255, 255, 240, 200))
        self.screen.blit(detail_bg, detail_rect)

        fnt_sm = pygame.font.SysFont(None, 24)
        hdr = fnt_sm.render("Connection Details (Paused)", True, (0, 0, 0))
        self.screen.blit(hdr, (detail_rect.x + 10, detail_rect.y + 10))

        line_y = detail_rect.y + 40
        shown_count = 0
        for p in self.people:
            if p.alive and (p.in_contract or p.in_cloud_contract):
                link_type = "CLOUD" if p.in_cloud_contract else "EDGE"
                conn_text = f"User@({int(p.gx)},{int(p.gy)}): {p.link_comm_type}, {p.link_fading_type}, {link_type}"
                speed_text = f"Down: {p.link_download_speed:.2f}Mbps / Up: {p.link_upload_speed:.2f}Mbps"
                r1 = fnt_sm.render(conn_text, True, (0, 0, 0))
                r2 = fnt_sm.render(speed_text, True, (0, 0, 0))
                self.screen.blit(r1, (detail_rect.x + 10, line_y))
                line_y += r1.get_height() + 2
                self.screen.blit(r2, (detail_rect.x + 10, line_y))
                line_y += r2.get_height() + 10
                shown_count += 1
                if line_y > detail_rect.y + detail_rect.height - 30:
                    break
        if shown_count == 0:
            no_conn = fnt_sm.render("No active connections.", True, (0, 0, 0))
            self.screen.blit(no_conn, (detail_rect.x + 10, line_y))

    def draw_right_panel(self):
        panel_rect = pygame.Rect(MAIN_AREA_WIDTH, 0, RIGHT_PANEL_WIDTH, SCREEN_HEIGHT - 300)
        panel_bg = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_bg.fill((240, 240, 255))
        self.screen.blit(panel_bg, panel_rect)

        fnt_sm = pygame.font.SysFont(None, 24)
        y_offset = 20

        title = fnt_sm.render("Edge Server Information", True, (0, 0, 0))
        self.screen.blit(title, (MAIN_AREA_WIDTH + 10, y_offset))
        y_offset += 30

        for i, (gx, gy, c, lbl, op) in enumerate(self.edge_servers_positions):
            info_box = pygame.Surface((RIGHT_PANEL_WIDTH - 20, 120))
            info_box.fill((255, 255, 255))

            texts = [
                f"Server: {lbl}",
                f"Operator: {op}",
                f"Location: ({gx}, {gy})",
                f"Load: {self.edge_qs[i]}/{EDGE_SERVER_MAX_Q}",
                f"Revenue: {to_musd(self.edge_revenues[i]):.0f}m$"
            ]

            for j, text in enumerate(texts):
                text_surf = fnt_sm.render(text, True, (0, 0, 0))
                info_box.blit(text_surf, (10, j * 20 + 10))

            self.screen.blit(info_box, (MAIN_AREA_WIDTH + 10, y_offset))
            y_offset += 130

    def draw_speed_buttons(self):
        fnt_sm = pygame.font.SysFont(None, 24)
        for label, pos, spd in self.speed_buttons:
            rect = pygame.Rect(pos[0], pos[1], 60, 30)
            pygame.draw.rect(self.screen, (200, 200, 200), rect)
            tr = fnt_sm.render(label, True, (0, 0, 0))
            self.screen.blit(tr, (rect.x + rect.width//2 - tr.get_width()//2, rect.y + rect.height//2 - tr.get_height()//2))

    def add_edge_server(self, gx, gy):
        if (gx, gy) in self.blocked_cells or (gx, gy) in self.building_set:
            self.log_event("Cannot place Edge Server on blocked or building cell.")
            return False
        color = (200, 50, 50)
        lbl = f"E{len(self.edge_servers_positions) + 1}"
        operator_name = "DefaultOperator"
        self.edge_servers_positions.append((gx, gy, color, lbl, operator_name))
        self.edge_qs.append(0)
        self.edge_revenues.append(0.0)
        self.edge_unit_prices.append(0.01)
        self.last_edge_prices.append(0.01)
        self.blocked_cells.add((gx, gy))
        self.log_event(f"Added Edge Server {lbl} at ({gx}, {gy})")
        return True

    def delete_edge_server(self, idx):
        if 0 <= idx < len(self.edge_servers_positions):
            lbl = self.edge_servers_positions[idx][3]
            gx = self.edge_servers_positions[idx][0]
            gy = self.edge_servers_positions[idx][1]
            for p in self.people:
                if p.link_target == lbl:
                    p.clear_contract()
                    p.state = "Idle"
            self.edge_servers_positions.pop(idx)
            self.edge_qs.pop(idx)
            self.edge_revenues.pop(idx)
            self.edge_unit_prices.pop(idx)
            self.last_edge_prices.pop(idx)
            self.blocked_cells.discard((gx, gy))
            self.log_event(f"Deleted Edge Server {lbl} at ({gx}, {gy})")
            return True
        else:
            self.log_event("Invalid Edge Server index for deletion.")
            return False

    def add_basestation(self, gx, gy):
        if (gx, gy) in self.basestations or (gx, gy) in self.building_set or (gx, gy) in self.blocked_cells:
            self.log_event("Cannot place Base Station on blocked or building cell.")
            return False
        self.basestations.append((gx, gy))
        self.decorations.append(('basestation', (gx, gy)))
        self.log_event(f"Added Base Station at ({gx}, {gy})")
        return True

    def delete_basestation(self, idx):
        if 0 <= idx < len(self.basestations):
            bx, by = self.basestations.pop(idx)
            for d in self.decorations:
                if d[0] == 'basestation' and d[1][0] == bx and d[1][1] == by:
                    self.decorations.remove(d)
                    break
            self.log_event(f"Deleted Base Station at ({bx}, {by})")
            return True
        else:
            self.log_event("Invalid Base Station index for deletion.")
            return False

    def increase_edge_server_capacity(self, idx):
        global EDGE_SERVER_MAX_Q
        EDGE_SERVER_MAX_Q = int(EDGE_SERVER_MAX_Q * 1.1)
        self.log_event(f"Increased capacity for Edge Server #{idx+1}. New MAX_Q={EDGE_SERVER_MAX_Q}")

    def set_edge_server_operator(self, idx):
        if 0 <= idx < len(self.edge_servers_positions):
            menu_options = [(op, f"op_{op}") for op in OPERATORS]
            self.menu = Menu([opt[0] for opt in menu_options], (MAIN_AREA_WIDTH - 200, 100), self.font)
            self.menu_actions = {opt[0]: opt[1] for opt in menu_options}
            self.current_edge_idx = idx

    def handle_menu_selection(self, selection, click_pos):
        action = self.menu_actions.get(selection, None)
        if action:
            if action.startswith("op_"):
                new_op = action[3:]
                if hasattr(self, 'current_edge_idx'):
                    idx = self.current_edge_idx
                    gx, gy, c, lbl, old_op = self.edge_servers_positions[idx]
                    self.edge_servers_positions[idx] = (gx, gy, c, lbl, new_op)
                    self.log_event(f"Changed {lbl} operator from {old_op} to {new_op}")
            elif action.startswith("delete_edge_"):
                idx = int(action.split("_")[-1])
                self.delete_edge_server(idx)
            elif action.startswith("delete_basestation_"):
                idx = int(action.split("_")[-1])
                self.delete_basestation(idx)
            elif action.startswith("expand_edge_"):
                idx = int(action.split("_")[-1])
                self.increase_edge_server_capacity(idx)
            elif action == "add_edge_server":
                gx, gy = self.screen_to_grid(click_pos)
                self.add_edge_server(gx, gy)
            elif action == "add_basestation":
                gx, gy = self.screen_to_grid(click_pos)
                self.add_basestation(gx, gy)

    def screen_to_grid(self, pos):
        ix, iy = pos
        x = (ix - ISO_OFFSETX) / ISO_SCALE
        y = (2 * iy - x) / 1.0
        gx = int(round((x + y) / 2 / BLOCK_SIZE - camera_offset_x / BLOCK_SIZE))
        gy = int(round((y - x / 2) / BLOCK_SIZE - camera_offset_y / BLOCK_SIZE))
        gx = max(0, min(GRID_W - 1, gx))
        gy = max(0, min(GRID_H - 1, gy))
        return gx, gy

    def get_menu_at_pos(self, pos):
        if self.menu and self.menu.visible:
            return self.menu
        return None

    def handle_event(self, event):
        global PAUSED, SIMULATION_SPEED, SHOW_CONNECTION_DETAILS, SELECTED_LINK
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == K_SPACE:
                PAUSED = not PAUSED
            elif event.key == K_c:
                if PAUSED:
                    SHOW_CONNECTION_DETAILS = not SHOW_CONNECTION_DETAILS
            elif event.key == K_1:
                SIMULATION_SPEED = 1
            elif event.key == K_2:
                SIMULATION_SPEED = 2
            elif event.key == K_3:
                SIMULATION_SPEED = 4
            elif event.key == K_4:
                SIMULATION_SPEED = 8
            elif event.key == K_5:
                SIMULATION_SPEED = 16
            elif event.key == K_6:
                SIMULATION_SPEED = 32
            elif event.key == K_7:
                SIMULATION_SPEED = 64
            elif event.key == K_8:
                SIMULATION_SPEED = 128

        if event.type == MOUSEBUTTONDOWN:
            if PAUSED:
                SELECTED_LINK = self.detect_link_click(event.pos)
            for label, pos, spd in self.speed_buttons:
                rect = pygame.Rect(pos[0], pos[1], 60, 30)
                if rect.collidepoint(event.pos):
                    if label == "Pause":
                        PAUSED = not PAUSED
                    else:
                        SIMULATION_SPEED = spd

            if self.menu and self.menu.visible:
                selection = self.menu.handle_event(event)
                if selection:
                    self.handle_menu_selection(selection, event.pos)

        if event.type == MOUSEWHEEL:
            global ISO_SCALE
            if event.y > 0:
                ISO_SCALE += 0.1
            elif event.y < 0:
                ISO_SCALE -= 0.1
            ISO_SCALE = max(0.1, min(ISO_SCALE, 2.0))

    def detect_link_click(self, mouse_pos):
        mx, my = mouse_pos
        best_dist = LINK_CLICK_DISTANCE_THRESHOLD
        selected = None
        for (start, end, person) in self.link_lines:
            dist = self.point_line_distance(mx, my, start, end)
            if dist < best_dist:
                best_dist = dist
                selected = (person, start, end)
        return selected

    def point_line_distance(self, px, py, A, B):
        Ax, Ay = A
        Bx, By = B
        ABx = Bx - Ax
        ABy = By - Ay
        APx = px - Ax
        APy = py - Ay
        ab_len_sq = ABx*ABx + ABy*ABy
        if ab_len_sq == 0:
            return math.sqrt((px - Ax)**2 + (py - Ay)**2)
        t = (APx*ABx + APy*ABy) / ab_len_sq
        if t < 0:
            return math.sqrt((px - Ax)**2 + (py - Ay)**2)
        elif t > 1:
            return math.sqrt((px - Bx)**2 + (py - By)**2)
        else:
            Cx = Ax + t*ABx
            Cy = Ay + t*ABy
            return math.sqrt((px - Cx)**2 + (py - Cy)**2)

    def draw_link_popup(self, link_info):
        person, start, end = link_info
        if not person.alive or not (person.in_contract or person.in_cloud_contract):
            return
        fnt_sm = pygame.font.SysFont(None, 24)
        popup_w = 300
        popup_h = 200
        mx, my = pygame.mouse.get_pos()
        popup_rect = pygame.Rect(mx, my, popup_w, popup_h)
        if popup_rect.right > SCREEN_WIDTH:
            popup_rect.x = SCREEN_WIDTH - popup_w
        if popup_rect.bottom > SCREEN_HEIGHT:
            popup_rect.y = SCREEN_HEIGHT - popup_h

        popup_bg = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
        popup_bg.fill((255, 255, 200, 200))
        self.screen.blit(popup_bg, (popup_rect.x, popup_rect.y))

        link_type = "CLOUD" if person.in_cloud_contract else "EDGE"
        lines = [
            f"Link Type: {link_type}",
            f"Comm Type: {person.link_comm_type}",
            f"Fading: {person.link_fading_type}",
            f"Path Loss: {person.link_path_loss_db:.2f} dB",
            f"Shadowing: {person.link_shadowing_db:.2f} dB",
            f"Rayleigh Factor: {person.rayleigh_factor:.2f}",
            f"Down: {person.link_download_speed:.2f} Mbps",
            f"Up: {person.link_upload_speed:.2f} Mbps"
        ]

        y_off = 10
        for ln in lines:
            r = fnt_sm.render(ln, True, (0,0,0))
            self.screen.blit(r, (popup_rect.x+10, popup_rect.y+y_off))
            y_off += r.get_height() + 2

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MEC Sim Enhanced with Visualized Techniques")
    clock = pygame.time.Clock()

    env = MECEnvironment(screen)

    running = True
    try:
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                env.handle_event(event)

            env.handle_input()
            env.step()
            env.draw()
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # 시뮬레이션 끝: 결과 저장 + 그래프 그리기(팝업)
        env.save_and_plot_data()
        del env
        pygame.quit()

if __name__ == "__main__":
    main()

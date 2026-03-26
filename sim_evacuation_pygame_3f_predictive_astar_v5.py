
"""
Dynamic Evacuation Simulation — 3F Predictive A* v5
보완 사항 (v4 → v5):
  1. 미대피 에이전트 방지: 막힌 노드에서도 relaxed 경로 확보 강화
  2. 계단 혼잡 완화: STAIR_CAPACITY 기반 동적 가중치 및 분산 라우팅
  3. EX_C 차단 이벤트 후 즉시 전체 재경로 트리거
  4. fallback 회복 조건 개선 (stalled_time < 1.5 → 0.8s)
  5. 에이전트 이질성 추가 (속도, 반응 지연, 노약자 그룹)
  6. 연기 확산 개선: 도어 차단 시 연기 감쇠 반영
  7. 통계 화면 보완: P50/P90/미대피 이유 표시
  8. 재경로 쿨다운 완화 및 improvement threshold 조정
  9. 데드락 에이전트 자동 이동 (stall > 12s → 인근 노드 강제 이동)
"""

import math
import json
import csv
import random
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import pygame
import networkx as nx

WIDTH, HEIGHT = 1560, 800
FPS = 60

BG = (245, 247, 250)
FLOOR_BG = (232, 236, 242)
EDGE_COLOR = (180, 188, 198)
NODE_COLOR = (90, 100, 115)
TEXT_COLOR = (40, 45, 52)
FIRE_COLOR = (225, 70, 45)
SMOKE_COLOR = (120, 120, 120)
HEAT_COLOR = (230, 120, 70)
PATH_COLOR = (40, 190, 90)
REROUTE_PATH_COLOR = (245, 185, 60)
EXIT_COLOR = (30, 120, 220)
STAIR_COLOR = (120, 70, 180)
AGENT_COLOR = (30, 140, 255)
DANGER_COLOR = (210, 40, 40)
BLOCK_COLOR = (120, 20, 20)
WALL_COLOR = (85, 25, 25)
FIRE_DOOR_OPEN_COLOR = (70, 130, 200)
FIRE_DOOR_CLOSED_COLOR = (240, 150, 40)
LEGEND_BG = (255, 255, 255)
ELDERLY_COLOR = (200, 140, 60)   # 노약자 에이전트
GUIDE_GO_COLOR = (50, 205, 110)      # 안전 방향(유도등 점등)
GUIDE_CAUTION_COLOR = (255, 205, 85) # 주의(혼잡/연기 증가)
GUIDE_STOP_COLOR = (220, 70, 70)     # 위험(진입 금지)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Dynamic Evacuation Simulation - 3F Predictive A* v5")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 16)
small_font = pygame.font.SysFont("arial", 12)
big_font = pygame.font.SysFont("arial", 22)

FLOORS = [3, 2, 1]
FLOOR_RECTS = {}
for i, floor in enumerate(FLOORS):
    FLOOR_RECTS[floor] = pygame.Rect(35, 35 + i * 225, 1460, 195)

# [보완 5] 에이전트 타입 정의
AGENT_TYPES = {
    "normal":  {"speed_base": 90,  "speed_var": 18, "reaction_delay": 0.0, "weight": 0.70},
    "elderly": {"speed_base": 55,  "speed_var": 8,  "reaction_delay": 1.5, "weight": 0.15},
    "child":   {"speed_base": 75,  "speed_var": 12, "reaction_delay": 0.5, "weight": 0.15},
}


@dataclass
class Agent:
    node: str
    x: float
    y: float
    speed: float = 90.0
    radius: int = 4
    agent_type: str = "normal"
    path: list = field(default_factory=list)
    path_index: int = 0
    current_edge: tuple = None
    progress: float = 0.0
    evacuated: bool = False
    reroute_flash: float = 0.0
    spawn_node: str = ""
    evacuation_time: float = None
    travelled_distance: float = 0.0
    cumulative_smoke: float = 0.0
    stalled_time: float = 0.0
    reroute_count: int = 0
    reached_exit: str = ""
    fallback_mode: bool = False
    deadlock_escape_count: int = 0
    route_hold_until: float = 0.0
    last_reroute_time: float = -999.0
    # [보완 5] 반응 지연 (이벤트 직후 잠깐 멈추는 효과)
    reaction_delay_remaining: float = 0.0
    # [보완 9] 데드락 강제 탈출 타이머
    hard_stall_time: float = 0.0
    stranded_reason: str = ""  # 미대피 이유 기록


class EvacuationSim:
    def __init__(self):
        self.base_graph = self.build_graph()
        self.G = self.base_graph.copy()
        self.setup_scenario(1)

    def build_graph(self):
        G = nx.Graph()

        x_positions = [120 + i * 120 for i in range(11)]
        y_map = {}
        for floor in FLOORS:
            rect = FLOOR_RECTS[floor]
            y_map[floor] = {
                "top": rect.y + 58,
                "bot": rect.y + 128,
                "mid": rect.y + 93,
            }

        stair_cols = [2, 6, 10]
        firedoor_cols = [4, 8]

        for floor in FLOORS:
            for i, x in enumerate(x_positions, start=1):
                G.add_node(f"F{floor}_T{i}", floor=floor, pos=(x, y_map[floor]["top"]), kind="normal")
                G.add_node(f"F{floor}_B{i}", floor=floor, pos=(x, y_map[floor]["bot"]), kind="normal")
            for idx in stair_cols:
                x = x_positions[idx - 1]
                G.add_node(f"F{floor}_S{idx}", floor=floor, pos=(x, y_map[floor]["mid"]), kind="stair")

        exits = {
            "EX_L": (1, 55, y_map[1]["mid"]),
            "EX_C": (1, x_positions[5], FLOOR_RECTS[1].y + 165),
            "EX_R": (1, 1450, y_map[1]["mid"]),
        }
        for name, (floor, x, y) in exits.items():
            G.add_node(name, floor=floor, pos=(x, y), kind="exit")

        def add_edge(a, b):
            ax, ay = G.nodes[a]["pos"]
            bx, by = G.nodes[b]["pos"]
            dist = math.hypot(ax - bx, ay - by)
            G.add_edge(a, b, length=dist, weight=dist, congestion=0)

        connectors = [1, 3, 6, 9, 11]
        for floor in FLOORS:
            for i in range(1, 11):
                add_edge(f"F{floor}_T{i}", f"F{floor}_T{i+1}")
                add_edge(f"F{floor}_B{i}", f"F{floor}_B{i+1}")
            for i in connectors:
                add_edge(f"F{floor}_T{i}", f"F{floor}_B{i}")
            for idx in stair_cols:
                add_edge(f"F{floor}_T{idx}", f"F{floor}_S{idx}")
                add_edge(f"F{floor}_B{idx}", f"F{floor}_S{idx}")

        for upper, lower in zip(FLOORS[:-1], FLOORS[1:]):
            for idx in stair_cols:
                add_edge(f"F{upper}_S{idx}", f"F{lower}_S{idx}")

        add_edge("F1_T1", "EX_L")
        add_edge("F1_B1", "EX_L")
        add_edge("F1_T11", "EX_R")
        add_edge("F1_B11", "EX_R")
        add_edge("F1_B6", "EX_C")
        add_edge("F1_T6", "EX_C")

        self.fire_door_edges = set()
        for floor in FLOORS:
            for col in firedoor_cols:
                self.fire_door_edges.add(tuple(sorted((f"F{floor}_T{col}", f"F{floor}_T{col+1}"))))
                self.fire_door_edges.add(tuple(sorted((f"F{floor}_B{col}", f"F{floor}_B{col+1}"))))
            for idx in stair_cols:
                self.fire_door_edges.add(tuple(sorted((f"F{floor}_T{idx}", f"F{floor}_S{idx}"))))
                self.fire_door_edges.add(tuple(sorted((f"F{floor}_B{idx}", f"F{floor}_S{idx}"))))
        return G

    def edge_key(self, a, b):
        return tuple(sorted((a, b)))

    def heuristic_to_exit(self, a, b):
        ax, ay = self.G.nodes[a]["pos"]
        bx, by = self.G.nodes[b]["pos"]
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def is_stair_edge(self, u, v):
        return self.G.nodes[u]["kind"] == "stair" and self.G.nodes[v]["kind"] == "stair"

    def is_fire_door_edge(self, u, v):
        return self.edge_key(u, v) in self.fire_door_edges

    def edge_is_blocked(self, u, v):
        key = self.edge_key(u, v)
        return key in self.blocked_edges_manual or key in self.blocked_edges_dynamic or self.G[u][v].get("weight", 1e9) >= 1e8

    def setup_scenario(self, scenario_id):
        self.scenario_id = scenario_id
        self.time_s = 0.0
        self.G = self.base_graph.copy()
        self.agents = []
        self.smoke = {n: 0.0 for n in self.G.nodes}
        self.temperature = {n: 22.0 for n in self.G.nodes}  # 센서 기반 온도(°C)
        self.fire = set()
        self.initial_fire = set()
        self.blocked_nodes = set()
        self.blocked_edges_manual = set()
        self.blocked_edges_dynamic = set()
        self.closed_fire_doors = set()
        self.reroute_events = 0
        self.max_congestion_by_edge = {}
        self.metrics_exported = False
        self.finished_reason = ""
        self.summary_cache = None
        self.max_sim_time = 180.0
        self.event_log = []
        self.scheduled_events = []
        self.triggered_event_ids = set()
        self.edge_guidance_state = {}
        # [보완 3] 이벤트 직후 전체 재경로 플래그
        self.force_recompute_all = False

        self.STAIR_CAPACITY = 12
        self.FIRE_DOOR_FLOW_CAPACITY = 4
        # [보완 8] improvement 임계값 완화 (0.84 → 0.88)
        self.REROUTE_IMPROVEMENT_RATIO = 0.88
        # [보완 8] 재경로 쿨다운 단축 (2.5 → 1.8s)
        self.MIN_REROUTE_INTERVAL = 1.8
        self.MIN_PATH_HOLD_TIME = 1.5
        self.PREDICTION_LOOKAHEAD = 3
        # [보완 4] fallback 임계값 유지, 회복 조건 강화
        self.FALLBACK_STALL_THRESHOLD = 7.0
        # [보완 9] 데드락 강제 탈출 임계값
        self.HARD_STALL_THRESHOLD = 12.0
        # 아이디어 반영: 센서 융합 위험도(연기/온도/혼잡) 가중치
        self.RISK_WEIGHTS = {"smoke": 0.52, "temperature": 0.28, "congestion": 0.20}

        scenarios = {
            1: {
                "fire": ["F2_T5"],
                "blocked_nodes": [],
                "blocked_edges": [],
                "events": [
                    {"id": "close_center_exit", "time": 12.0, "type": "block_node", "node": "EX_C", "label": "Central exit EX_C closed"},
                    {"id": "close_mid_stair", "time": 18.0, "type": "block_edge", "edge": ("F3_T6", "F3_S6"), "label": "3F central stair lobby blocked"},
                ],
                "name": "Scenario 1: 3F standard building / 2F corridor fire",
            },
            2: {
                "fire": ["F3_B8"],
                "blocked_nodes": [],
                "blocked_edges": [],
                "events": [
                    {"id": "close_right_exit", "time": 10.0, "type": "block_node", "node": "EX_R", "label": "Right exit closed"},
                    {"id": "close_upper_right_stair", "time": 20.0, "type": "block_edge", "edge": ("F3_B10", "F3_S10"), "label": "3F right stair lobby blocked"},
                ],
                "name": "Scenario 2: 3F standard building / 3F right-wing fire",
            },
            3: {
                "fire": ["F1_S2"],
                "blocked_nodes": [],
                "blocked_edges": [("F1_T2", "F1_S2"), ("F1_B2", "F1_S2")],
                "events": [
                    {"id": "close_left_exit", "time": 11.0, "type": "block_node", "node": "EX_L", "label": "Left exit closed"},
                ],
                "name": "Scenario 3: 3F standard building / 1F left stair fire",
            },
        }

        sc = scenarios[scenario_id]
        self.fire = set(sc["fire"])
        self.initial_fire = set(sc["fire"])
        self.blocked_nodes = set(sc["blocked_nodes"])
        self.blocked_edges_manual = {tuple(sorted(e)) for e in sc["blocked_edges"]}
        self.scheduled_events = sc.get("events", [])
        self.scenario_name = sc["name"]

        self.spawn_agents()
        self.update_hazards()
        self.recompute_paths_all()

    def spawn_agents(self):
        """[보완 5] 에이전트 타입 다양화 (normal/elderly/child 혼합)"""
        counts = {}
        for floor in FLOORS:
            for i in range(2, 11):
                node_t = f"F{floor}_T{i}"
                node_b = f"F{floor}_B{i}"
                if node_t not in self.initial_fire:
                    counts[node_t] = 3 if floor == 3 else 2
                if node_b not in self.initial_fire:
                    counts[node_b] = 2

        for fire_node in self.initial_fire:
            counts.pop(fire_node, None)
            if fire_node in self.G:
                for nb in self.G.neighbors(fire_node):
                    if nb in counts:
                        counts[nb] = max(0, counts[nb] - 1)

        rng = random.Random(42)  # 재현 가능한 난수
        type_pool = []
        for atype, cfg in AGENT_TYPES.items():
            type_pool.append((atype, cfg["weight"]))

        def pick_type():
            r = rng.random()
            cum = 0
            for atype, w in type_pool:
                cum += w
                if r <= cum:
                    return atype
            return "normal"

        for node, count in counts.items():
            x, y = self.G.nodes[node]["pos"]
            for k in range(count):
                atype = pick_type()
                cfg = AGENT_TYPES[atype]
                spd = cfg["speed_base"] + rng.randint(0, cfg["speed_var"])
                jx = x + (hash((node, k, 'x')) % 19 - 9)
                jy = y + (hash((node, k, 'y')) % 19 - 9)
                self.agents.append(
                    Agent(
                        node=node,
                        x=jx,
                        y=jy,
                        speed=float(spd),
                        agent_type=atype,
                        radius=4 if atype != "elderly" else 5,
                        spawn_node=node,
                        reaction_delay_remaining=cfg["reaction_delay"],
                    )
                )

    def process_scheduled_events(self):
        for ev in self.scheduled_events:
            if ev["id"] in self.triggered_event_ids:
                continue
            if self.time_s < ev["time"]:
                continue
            if ev["type"] == "block_node":
                self.blocked_nodes.add(ev["node"])
            elif ev["type"] == "block_edge":
                self.blocked_edges_manual.add(tuple(sorted(ev["edge"])))
            self.triggered_event_ids.add(ev["id"])
            self.event_log.append((self.time_s, ev["label"]))
            # [보완 3] 이벤트 발생 시 즉시 전체 재경로 플래그 설정
            self.force_recompute_all = True
            # [보완 5] 반응 지연 재설정
            for ag in self.agents:
                if not ag.evacuated:
                    ag.reaction_delay_remaining = max(
                        ag.reaction_delay_remaining,
                        AGENT_TYPES[ag.agent_type]["reaction_delay"]
                    )

    def spread_fire(self):
        if len(self.fire) >= 5:
            return
        if self.time_s > 0 and int(self.time_s) % 28 == 0 and abs(self.time_s - round(self.time_s)) < 0.05:
            candidates = []
            for f in list(self.fire):
                for nb in self.G.neighbors(f):
                    if nb in self.fire or nb in self.blocked_nodes:
                        continue
                    if self.G.nodes[nb]["kind"] == "exit":
                        continue
                    candidates.append(nb)
            if candidates:
                self.fire.add(sorted(candidates)[0])

    def build_traversable_graph(self):
        H = nx.Graph()
        for n, d in self.G.nodes(data=True):
            if n not in self.blocked_nodes:
                H.add_node(n, **d)
        for u, v, data in self.G.edges(data=True):
            key = self.edge_key(u, v)
            if u in self.blocked_nodes or v in self.blocked_nodes:
                continue
            if key in self.blocked_edges_manual or key in self.blocked_edges_dynamic:
                continue
            if data.get("weight", 1e9) >= 1e8:
                continue
            H.add_edge(u, v, **dict(data))
        return H

    def build_relaxed_graph(self):
        """[보완 1] relaxed 그래프: 연기 구간도 포함하되 높은 페널티"""
        H = nx.Graph()
        for n, d in self.G.nodes(data=True):
            if n not in self.blocked_nodes:
                H.add_node(n, **d)
        for u, v, data in self.G.edges(data=True):
            if u in self.blocked_nodes or v in self.blocked_nodes:
                continue
            key = self.edge_key(u, v)
            if key in self.blocked_edges_manual:
                continue
            if u in self.fire or v in self.fire:
                continue
            weight = data.get("weight", data["length"])
            smoke = max(self.smoke.get(u, 0.0), self.smoke.get(v, 0.0))
            if key in self.blocked_edges_dynamic or weight >= 1e8:
                # [보완 1] 동적 차단 구간: 페널티 줄여서 탈출 가능하게
                weight = data["length"] + 1800 + 1200 * smoke
            if self.is_fire_door_edge(u, v) and key in self.closed_fire_doors:
                weight += 180
            edge_data = dict(data)
            edge_data["weight"] = weight
            H.add_edge(u, v, **edge_data)
        return H

    def path_cost(self, path):
        if not path or len(path) < 2:
            return float("inf")
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            if not self.G.has_edge(a, b):
                return float("inf")
            total += self.G[a][b].get("weight", 1e9)
        return total

    def routing_cost(self, path):
        if not path or len(path) < 2:
            return float("inf")
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            if not self.G.has_edge(a, b):
                return float("inf")
            data = self.G[a][b]
            dist = data["length"]
            congestion = data.get("congestion", 0)
            smoke = max(self.smoke.get(a, 0.0), self.smoke.get(b, 0.0))
            edge_temp = max(self.temperature.get(a, 22.0), self.temperature.get(b, 22.0))
            temp_norm = min(1.0, max(0.0, (edge_temp - 22.0) / 58.0))
            cost = dist + 320 * smoke + 110 * temp_norm + 120 * congestion
            if self.is_stair_edge(a, b):
                overflow = max(0, congestion - self.STAIR_CAPACITY)
                # [보완 2] 계단 overflow 페널티 강화
                cost += 320 * overflow + 100
            if self.is_fire_door_edge(a, b) and self.edge_key(a, b) in self.closed_fire_doors:
                cost += 160 + 45 * congestion
            total += cost
        return total

    def predicted_inflow(self, u, v):
        edge = self.edge_key(u, v)
        inflow = 0
        for ag in self.agents:
            if ag.evacuated or not ag.path:
                continue
            remain = ag.path[ag.path_index:]
            if len(remain) < 2:
                continue
            for a, b in zip(remain[:-1], remain[1:]):
                if self.edge_key(a, b) == edge:
                    inflow += 1
                    break
        return inflow

    def predictive_edge_cost(self, a, b, depth=0):
        if not self.G.has_edge(a, b):
            return float("inf")
        data = self.G[a][b]
        dist = data["length"]
        congestion = data.get("congestion", 0)
        smoke = max(self.smoke.get(a, 0.0), self.smoke.get(b, 0.0))
        edge_temp = max(self.temperature.get(a, 22.0), self.temperature.get(b, 22.0))
        temp_norm = min(1.0, max(0.0, (edge_temp - 22.0) / 58.0))
        inflow = self.predicted_inflow(a, b)

        cost = dist + 340 * smoke + 100 * temp_norm + 120 * congestion
        cost += (110 - 20 * min(depth, 3)) * inflow

        if self.is_stair_edge(a, b):
            overflow = max(0, congestion + inflow - self.STAIR_CAPACITY)
            # [보완 2] 예측 단계에서도 overflow 페널티 강화
            cost += 380 * overflow + 140

        if self.is_fire_door_edge(a, b) and self.edge_key(a, b) in self.closed_fire_doors:
            cost += 180 + 60 * (congestion + inflow)

        if self.edge_is_blocked(a, b):
            return float("inf")
        return cost

    def predictive_path_cost(self, path):
        if not path or len(path) < 2:
            return float("inf")
        total = 0.0
        for depth, (a, b) in enumerate(zip(path[:-1], path[1:])):
            c = self.predictive_edge_cost(a, b, depth=depth)
            if c == float("inf"):
                return c
            total += c
        return total

    def update_hazards(self):
        self.process_scheduled_events()
        self.spread_fire()

        if self.time_s >= 8.0:
            self.closed_fire_doors = set(self.fire_door_edges)

        # [보완 6] 연기 확산: 방화문 차단 시 감쇠 강화
        smoke_graph = nx.Graph()
        for n, d in self.G.nodes(data=True):
            if n not in self.blocked_nodes:
                smoke_graph.add_node(n, **d)

        for u, v, data in self.G.edges(data=True):
            key = self.edge_key(u, v)
            if u in self.blocked_nodes or v in self.blocked_nodes:
                continue
            if key in self.blocked_edges_manual:
                continue
            smoke_length = data["length"]
            if self.is_fire_door_edge(u, v) and key in self.closed_fire_doors:
                # [보완 6] 방화문 닫힘 시 연기 차단 효과를 더 강하게 반영
                smoke_length *= 5.0
            smoke_graph.add_edge(u, v, length=smoke_length)

        for n in self.G.nodes:
            if n in self.fire:
                self.smoke[n] = 1.0
                self.temperature[n] = 250.0
                continue
            if n in self.blocked_nodes:
                self.smoke[n] = 0.0
                self.temperature[n] = 22.0
                continue
            try:
                min_d = min(
                    nx.shortest_path_length(smoke_graph, source=f, target=n, weight="length")
                    for f in self.fire if f in smoke_graph and n in smoke_graph
                )
            except Exception:
                min_d = 10000
            front = max(0.0, self.time_s * 22 - min_d)
            self.smoke[n] = max(0.0, min(1.0, front / 400))
            # 센서 온도 모델: 화원과 거리, 시간에 따라 증가(완만한 포화 형태)
            temp_gain = max(0.0, (280 - min_d) / 280)
            self.temperature[n] = 22.0 + 58.0 * temp_gain * min(1.0, self.time_s / 25.0)

        self.blocked_edges_dynamic = set()

        for u, v, data in self.G.edges(data=True):
            key = self.edge_key(u, v)
            base = data["length"]

            if u in self.blocked_nodes or v in self.blocked_nodes:
                data["weight"] = 1e9
                self.blocked_edges_dynamic.add(key)
                continue
            if key in self.blocked_edges_manual:
                data["weight"] = 1e9
                self.blocked_edges_dynamic.add(key)
                continue

            smoke = max(self.smoke[u], self.smoke[v])

            if self.is_fire_door_edge(u, v):
                if u in self.fire or v in self.fire or smoke >= 0.99:
                    data["weight"] = 1e9
                    self.blocked_edges_dynamic.add(key)
                    continue
            else:
                if u in self.fire or v in self.fire or smoke >= 0.97:
                    data["weight"] = 1e9
                    self.blocked_edges_dynamic.add(key)
                    continue

            try:
                min_d = min(
                    min(
                        nx.shortest_path_length(self.G, source=f, target=u, weight="length"),
                        nx.shortest_path_length(self.G, source=f, target=v, weight="length"),
                    ) for f in self.fire
                )
            except Exception:
                min_d = 10000

            fire_penalty = (320 - min_d) * 7 if min_d < 320 else 0
            congestion = data.get("congestion", 0)
            if self.is_stair_edge(u, v):
                overflow = max(0, congestion - self.STAIR_CAPACITY)
                # [보완 2] 계단 overflow 가중치 강화
                congestion_penalty = congestion * 80 + overflow * 350
            else:
                congestion_penalty = congestion * 60
            smoke_penalty = smoke * 1500
            edge_temp = max(self.temperature.get(u, 22.0), self.temperature.get(v, 22.0))
            temp_penalty = max(0.0, edge_temp - 35.0) * 7.5
            door_penalty = 0
            if self.is_fire_door_edge(u, v) and key in self.closed_fire_doors:
                door_penalty = 170 + 50 * congestion

            data["weight"] = base + fire_penalty + congestion_penalty + smoke_penalty + temp_penalty + door_penalty

    def edge_risk_score(self, u, v):
        """센서(연기/온도/혼잡) 융합 위험도 [0,1]."""
        if not self.G.has_edge(u, v):
            return 1.0
        smoke = max(self.smoke.get(u, 0.0), self.smoke.get(v, 0.0))
        temp = max(self.temperature.get(u, 22.0), self.temperature.get(v, 22.0))
        temp_norm = min(1.0, max(0.0, (temp - 22.0) / 58.0))
        congestion = self.G[u][v].get("congestion", 0)
        congestion_norm = min(1.0, congestion / max(1, self.STAIR_CAPACITY))
        risk = (
            self.RISK_WEIGHTS["smoke"] * smoke
            + self.RISK_WEIGHTS["temperature"] * temp_norm
            + self.RISK_WEIGHTS["congestion"] * congestion_norm
        )
        return min(1.0, max(0.0, risk))

    def update_guidance_lights(self):
        """
        빛 기반 동적 유도 상태를 에지별로 업데이트.
        - go: 안전(초록)
        - caution: 주의(노랑)
        - stop: 위험/차단(빨강 또는 소등 대상)
        """
        states = {}
        for u, v in self.G.edges:
            key = self.edge_key(u, v)
            if self.edge_is_blocked(u, v):
                states[key] = "stop"
                continue
            risk = self.edge_risk_score(u, v)
            if risk >= 0.75:
                states[key] = "stop"
            elif risk >= 0.45:
                states[key] = "caution"
            else:
                states[key] = "go"
        self.edge_guidance_state = states

    def safest_exit_path(self, source, relaxed=False):
        """[보완 1] 항상 적어도 하나의 경로를 반환하도록 2단계 fallback"""
        exits = [n for n, d in self.G.nodes(data=True) if d["kind"] == "exit" and n not in self.blocked_nodes]
        if source in self.blocked_nodes:
            return None
        H = self.build_relaxed_graph() if relaxed else self.build_traversable_graph()

        best_path = None
        best_cost = float("inf")
        for ex in exits:
            try:
                path = nx.astar_path(
                    H,
                    source,
                    ex,
                    heuristic=lambda n, goal=ex: self.heuristic_to_exit(n, goal),
                    weight="weight"
                )
                cost = nx.path_weight(H, path, weight="weight")
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            except Exception:
                pass

        # [보완 1] 일반 경로 없으면 relaxed로 2차 시도
        if best_path is None and not relaxed:
            return self.safest_exit_path(source, relaxed=True)

        return best_path

    def recompute_paths_all(self):
        for ag in self.agents:
            if ag.evacuated:
                continue
            if self.G.nodes[ag.node]["kind"] == "exit" and ag.node not in self.blocked_nodes:
                ag.evacuated = True
                if ag.evacuation_time is None:
                    ag.evacuation_time = self.time_s
                    ag.reached_exit = ag.node
                continue

            use_relaxed = ag.stalled_time >= 5.0

            old_path = ag.path[ag.path_index:] if ag.path else []
            old_cost = self.routing_cost(old_path) if old_path else float("inf")

            if ag.current_edge is not None:
                u, v = ag.current_edge
                tail = self.safest_exit_path(v, relaxed=use_relaxed)
                cand = [u] + tail if tail else [u, v]
                new_cost = self.routing_cost(cand)
                # [보완 8] improvement ratio 완화
                if tail and new_cost < old_cost * self.REROUTE_IMPROVEMENT_RATIO:
                    ag.path = cand
                    ag.path_index = 0
                    ag.reroute_flash = 2.0
                    ag.reroute_count += 1
                    self.reroute_events += 1
                continue

            new_path = self.safest_exit_path(ag.node, relaxed=use_relaxed)
            if new_path is None or len(new_path) < 2:
                ag.path = []
                ag.path_index = 0
                continue

            changed = tuple(new_path) != tuple(old_path)
            new_cost = self.routing_cost(new_path)
            if (not old_path) or (changed and new_cost < old_cost * self.REROUTE_IMPROVEMENT_RATIO):
                ag.path = new_path
                ag.path_index = 0
                if changed and old_path:
                    ag.reroute_flash = 2.0
                    ag.reroute_count += 1
                    self.reroute_events += 1
                if use_relaxed:
                    ag.fallback_mode = True
                    ag.deadlock_escape_count += 1

    def update_congestion(self):
        counts = defaultdict(int)
        for ag in self.agents:
            if ag.evacuated:
                continue
            if ag.current_edge is not None:
                counts[self.edge_key(*ag.current_edge)] += 1
            elif ag.path and ag.path_index + 1 < len(ag.path):
                counts[self.edge_key(ag.node, ag.path[ag.path_index + 1])] += 1
        for u, v in self.G.edges:
            key = self.edge_key(u, v)
            c = counts.get(key, 0)
            self.G[u][v]["congestion"] = c
            self.max_congestion_by_edge[key] = max(self.max_congestion_by_edge.get(key, 0), c)

    def maybe_finish(self):
        if self.finished_reason:
            return
        if all(a.evacuated for a in self.agents):
            self.finished_reason = "all_evacuated"
        elif self.time_s >= self.max_sim_time:
            # 미대피 에이전트 이유 기록
            for ag in self.agents:
                if not ag.evacuated and not ag.stranded_reason:
                    if not ag.path:
                        ag.stranded_reason = "no_path"
                    elif ag.hard_stall_time >= self.HARD_STALL_THRESHOLD:
                        ag.stranded_reason = "hard_deadlock"
                    else:
                        ag.stranded_reason = "time_limit"
            remaining_paths = sum(1 for a in self.agents if (not a.evacuated) and a.path)
            self.finished_reason = "time_limit_with_active_paths" if remaining_paths else "time_limit_no_remaining_paths"

    def export_results(self, prefix=None):
        summary = self.build_summary()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = prefix or f"evac_results_s{self.scenario_id}_{stamp}"
        summary_path = f"{base}_summary.json"
        agents_path = f"{base}_agents.csv"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(agents_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "agent_id", "agent_type", "spawn_node", "final_node", "evacuated",
                "evacuation_time_s", "reached_exit", "travelled_distance_px",
                "cumulative_smoke", "stalled_time_s", "reroute_count",
                "deadlock_escape_count", "stranded_reason"
            ])
            for idx, a in enumerate(self.agents):
                writer.writerow([
                    idx, a.agent_type, a.spawn_node, a.node, a.evacuated,
                    round(a.evacuation_time, 3) if a.evacuation_time is not None else "",
                    a.reached_exit,
                    round(a.travelled_distance, 3),
                    round(a.cumulative_smoke, 3),
                    round(a.stalled_time, 3),
                    a.reroute_count,
                    a.deadlock_escape_count,
                    a.stranded_reason,
                ])
        self.metrics_exported = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved: {summary_path}")
        print(f"Saved: {agents_path}")

    def build_summary(self):
        evac_times = sorted([a.evacuation_time for a in self.agents if a.evacuation_time is not None])
        reroute_vals = [a.reroute_count for a in self.agents]
        stalled_vals = [a.stalled_time for a in self.agents]
        smoke_vals = [a.cumulative_smoke for a in self.agents]
        dist_vals = [a.travelled_distance for a in self.agents]
        temp_vals = list(self.temperature.values())
        exit_counts = {}
        for a in self.agents:
            if a.reached_exit:
                exit_counts[a.reached_exit] = exit_counts.get(a.reached_exit, 0) + 1

        stranded_reasons = {}
        for a in self.agents:
            if not a.evacuated and a.stranded_reason:
                stranded_reasons[a.stranded_reason] = stranded_reasons.get(a.stranded_reason, 0) + 1

        def pct(vals, p):
            if not vals:
                return None
            idx = min(len(vals) - 1, int((len(vals) - 1) * p))
            return vals[idx]

        top_edges = sorted(self.max_congestion_by_edge.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_edges_fmt = [{"edge": list(edge), "max_congestion": c} for edge, c in top_edges]

        # 에이전트 타입별 통계
        type_stats = {}
        for atype in AGENT_TYPES:
            group = [a for a in self.agents if a.agent_type == atype]
            if not group:
                continue
            gt = sorted([a.evacuation_time for a in group if a.evacuation_time is not None])
            type_stats[atype] = {
                "total": len(group),
                "evacuated": sum(a.evacuated for a in group),
                "avg_evac_time_s": round(sum(gt) / len(gt), 3) if gt else None,
            }

        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "finished_reason": self.finished_reason or "running",
            "sim_time_s": round(self.time_s, 3),
            "floors": len(FLOORS),
            "total_agents": len(self.agents),
            "evacuated_agents": sum(a.evacuated for a in self.agents),
            "stranded_agents": sum(not a.evacuated for a in self.agents),
            "evacuation_rate": round(sum(a.evacuated for a in self.agents) / max(1, len(self.agents)), 4),
            "avg_evacuation_time_s": round(sum(evac_times) / len(evac_times), 3) if evac_times else None,
            "p50_evacuation_time_s": round(pct(evac_times, 0.50), 3) if evac_times else None,
            "p90_evacuation_time_s": round(pct(evac_times, 0.90), 3) if evac_times else None,
            "max_evacuation_time_s": round(max(evac_times), 3) if evac_times else None,
            "avg_reroutes_per_agent": round(sum(reroute_vals) / len(reroute_vals), 3) if reroute_vals else 0.0,
            "agents_with_5plus_reroutes": sum(1 for v in reroute_vals if v >= 5),
            "agents_with_10plus_reroutes": sum(1 for v in reroute_vals if v >= 10),
            "max_reroutes_single_agent": max(reroute_vals) if reroute_vals else 0,
            "agents_using_fallback": sum(1 for a in self.agents if a.deadlock_escape_count > 0),
            "total_deadlock_escapes": sum(a.deadlock_escape_count for a in self.agents),
            "total_reroutes": self.reroute_events,
            "avg_smoke_exposure": round(sum(smoke_vals) / len(smoke_vals), 3) if smoke_vals else 0.0,
            "max_smoke_exposure": round(max(smoke_vals), 3) if smoke_vals else 0.0,
            "avg_temperature_c": round(sum(temp_vals) / len(temp_vals), 2) if temp_vals else 22.0,
            "max_temperature_c": round(max(temp_vals), 2) if temp_vals else 22.0,
            "avg_stalled_time_s": round(sum(stalled_vals) / len(stalled_vals), 3) if stalled_vals else 0.0,
            "max_stalled_time_s": round(max(stalled_vals), 3) if stalled_vals else 0.0,
            "avg_distance_px": round(sum(dist_vals) / len(dist_vals), 2) if dist_vals else 0.0,
            "max_distance_px": round(max(dist_vals), 2) if dist_vals else 0.0,
            "active_fire_nodes_final": len(self.fire),
            "blocked_corridors_final": len(self.blocked_edges_dynamic),
            "closed_fire_doors_final": len(self.closed_fire_doors),
            "fire_door_edges_total": len(self.fire_door_edges),
            "blocked_nodes_final": sorted(list(self.blocked_nodes)),
            "exit_usage": exit_counts,
            "top_congested_edges": top_edges_fmt,
            "events_triggered": [{"time_s": round(t, 3), "label": label} for t, label in self.event_log],
            "stranded_reasons": stranded_reasons,
            "agent_type_stats": type_stats,
        }

    def step(self, dt):
        if self.finished_reason:
            return

        self.time_s += dt
        self.update_congestion()
        self.update_hazards()
        self.update_guidance_lights()

        # [보완 3] 이벤트 발생 시 즉시 전체 재경로
        if self.force_recompute_all:
            self.recompute_paths_all()
            self.force_recompute_all = False
        elif int(self.time_s * 4) != int((self.time_s - dt) * 4):
            self.recompute_paths_all()

        door_entries_this_frame = {}
        stair_entries_this_frame = {}

        for ag in self.agents:
            if ag.reroute_flash > 0:
                ag.reroute_flash = max(0.0, ag.reroute_flash - dt)
            if ag.evacuated:
                continue

            # [보완 5] 반응 지연 처리
            if ag.reaction_delay_remaining > 0:
                ag.reaction_delay_remaining -= dt
                ag.stalled_time += dt
                continue

            ag.cumulative_smoke += self.smoke.get(ag.node, 0.0) * dt * 0.35

            if self.G.nodes[ag.node]["kind"] == "exit" and ag.node not in self.blocked_nodes:
                ag.evacuated = True
                if ag.evacuation_time is None:
                    ag.evacuation_time = self.time_s
                    ag.reached_exit = ag.node
                continue

            if not ag.path or ag.path_index + 1 >= len(ag.path):
                ag.stalled_time += dt
                ag.hard_stall_time += dt
                # [보완 9] 경로 없을 때 강제 재탐색
                if ag.hard_stall_time >= 5.0:
                    new_path = self.safest_exit_path(ag.node, relaxed=True)
                    if new_path and len(new_path) >= 2:
                        ag.path = new_path
                        ag.path_index = 0
                        ag.hard_stall_time = 0.0
                        ag.fallback_mode = True
                        ag.deadlock_escape_count += 1
                continue

            target = ag.path[ag.path_index + 1]
            key = self.edge_key(ag.node, target)

            blocked = self.edge_is_blocked(ag.node, target)
            if ag.fallback_mode and key not in self.blocked_edges_manual and ag.node not in self.fire and target not in self.fire:
                blocked = False

            if ag.current_edge is None and blocked:
                new_path = self.safest_exit_path(ag.node, relaxed=(ag.stalled_time >= self.FALLBACK_STALL_THRESHOLD))
                if new_path is None or len(new_path) < 2:
                    ag.path = []
                    ag.path_index = 0
                    ag.stalled_time += dt
                    ag.hard_stall_time += dt
                    continue
                ag.path = new_path
                ag.path_index = 0
                target = ag.path[ag.path_index + 1]
                key = self.edge_key(ag.node, target)

            if (not ag.fallback_mode) and ag.current_edge is None and self.is_fire_door_edge(ag.node, target) and key in self.closed_fire_doors:
                used = door_entries_this_frame.get(key, 0)
                if used >= self.FIRE_DOOR_FLOW_CAPACITY:
                    ag.stalled_time += dt
                    continue
                door_entries_this_frame[key] = used + 1

            if (not ag.fallback_mode) and ag.current_edge is None and self.is_stair_edge(ag.node, target):
                used = stair_entries_this_frame.get(key, 0)
                # [보완 2] 계단 진입 cap 조정 (더 많은 에이전트 동시 이동 허용)
                frame_cap = max(2, self.STAIR_CAPACITY // 2)
                if used >= frame_cap:
                    ag.stalled_time += dt
                    continue
                stair_entries_this_frame[key] = used + 1

            edge_len = self.G[ag.node][target]["length"]
            congestion = self.G[ag.node][target].get("congestion", 0)
            danger = min(1.0, max(self.smoke[ag.node], self.smoke[target]))
            speed = ag.speed * (1 - 0.34 * danger) / (1 + 0.14 * max(0, congestion - 1))

            if self.is_stair_edge(ag.node, target):
                overflow = max(0, congestion - self.STAIR_CAPACITY)
                speed /= (1 + 0.08 * overflow)
            if self.is_fire_door_edge(ag.node, target) and key in self.closed_fire_doors:
                speed *= 0.6
            if ag.fallback_mode:
                speed *= 1.15
            # [보완 5] 노약자는 추가 감속
            if ag.agent_type == "elderly":
                speed *= 0.85

            speed = max(14.0, speed)
            delta_progress = speed * dt / edge_len

            ag.current_edge = (ag.node, target)
            ag.progress += delta_progress
            ag.travelled_distance += min(delta_progress, 1.0) * edge_len
            ag.cumulative_smoke += ((self.smoke[ag.node] + self.smoke[target]) * 0.5) * dt

            x1, y1 = self.G.nodes[ag.node]["pos"]
            x2, y2 = self.G.nodes[target]["pos"]
            ag.x = x1 + (x2 - x1) * ag.progress
            ag.y = y1 + (y2 - y1) * ag.progress

            if delta_progress < 0.01:
                ag.stalled_time += dt
                ag.hard_stall_time += dt
            else:
                ag.stalled_time = max(0.0, ag.stalled_time - 0.5 * dt)
                ag.hard_stall_time = max(0.0, ag.hard_stall_time - dt)

            # [보완 4] fallback 회복 조건 강화 (0.8s 미만이면 해제)
            if ag.fallback_mode:
                if ag.stalled_time < 0.8:
                    ag.fallback_mode = False

            if ag.stalled_time >= self.FALLBACK_STALL_THRESHOLD:
                ag.fallback_mode = True

            if ag.progress >= 1.0:
                ag.node = target
                ag.x, ag.y = self.G.nodes[ag.node]["pos"]
                ag.progress = 0.0
                ag.current_edge = None
                ag.path_index += 1
                ag.hard_stall_time = 0.0  # 노드 이동 성공 시 리셋
                if ag.node in self.fire:
                    ag.path = []
                    ag.path_index = 0
                if self.G.nodes[ag.node]["kind"] == "exit" and ag.node not in self.blocked_nodes:
                    ag.evacuated = True
                    if ag.evacuation_time is None:
                        ag.evacuation_time = self.time_s
                        ag.reached_exit = ag.node

        self.maybe_finish()
        if self.finished_reason and not self.metrics_exported:
            self.export_results()

    def draw_block_marker(self, x1, y1, x2, y2):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        px, py = -dy / length, dx / length
        size = 8
        p1 = (mx - px * size, my - py * size)
        p2 = (mx + px * size, my + py * size)
        pygame.draw.line(screen, WALL_COLOR, p1, p2, 5)
        pygame.draw.line(screen, BLOCK_COLOR, (mx - size, my - size), (mx + size, my + size), 2)
        pygame.draw.line(screen, BLOCK_COLOR, (mx - size, my + size), (mx + size, my - size), 2)

    def draw(self, screen):
        screen.fill(BG)

        for floor, rect in FLOOR_RECTS.items():
            pygame.draw.rect(screen, FLOOR_BG, rect, border_radius=10)
            pygame.draw.rect(screen, (200, 208, 218), rect, width=2, border_radius=10)
            label = big_font.render(f"Floor {floor}", True, TEXT_COLOR)
            screen.blit(label, (rect.x + 10, rect.y + 8))

        for u, v, data in self.G.edges(data=True):
            x1, y1 = self.G.nodes[u]["pos"]
            x2, y2 = self.G.nodes[v]["pos"]
            key = self.edge_key(u, v)

            if key in self.blocked_edges_dynamic or key in self.blocked_edges_manual:
                pygame.draw.line(screen, DANGER_COLOR, (x1, y1), (x2, y2), 5)
                self.draw_block_marker(x1, y1, x2, y2)
                continue

            guide = self.edge_guidance_state.get(key)
            if guide == "go":
                guide_color = GUIDE_GO_COLOR
            elif guide == "caution":
                guide_color = GUIDE_CAUTION_COLOR
            elif guide == "stop":
                guide_color = GUIDE_STOP_COLOR
            else:
                guide_color = None

            if key in self.fire_door_edges:
                color = FIRE_DOOR_CLOSED_COLOR if key in self.closed_fire_doors else FIRE_DOOR_OPEN_COLOR
                if guide_color:
                    color = tuple(int(0.6 * c + 0.4 * g) for c, g in zip(color, guide_color))
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 4)
            else:
                smoke = max(self.smoke[u], self.smoke[v])
                heat = min(1.0, max(0.0, (max(self.temperature[u], self.temperature[v]) - 22.0) / 58.0))
                color = (
                    int(EDGE_COLOR[0] * (1 - smoke) + SMOKE_COLOR[0] * smoke),
                    int(EDGE_COLOR[1] * (1 - smoke) + SMOKE_COLOR[1] * smoke),
                    int(EDGE_COLOR[2] * (1 - smoke) + SMOKE_COLOR[2] * smoke),
                )
                color = (
                    int(color[0] * (1 - 0.35 * heat) + HEAT_COLOR[0] * (0.35 * heat)),
                    int(color[1] * (1 - 0.35 * heat) + HEAT_COLOR[1] * (0.35 * heat)),
                    int(color[2] * (1 - 0.35 * heat) + HEAT_COLOR[2] * (0.35 * heat)),
                )
                if guide_color:
                    color = tuple(int(0.55 * c + 0.45 * g) for c, g in zip(color, guide_color))
                congestion = data.get("congestion", 0)
                # [보완 2] 혼잡 시 에지 두께 강조
                width = 2 + min(6, congestion // 2)
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), width)

        path_edges = set()
        reroute_edges = set()
        for ag in self.agents:
            if ag.evacuated or not ag.path:
                continue
            target = reroute_edges if ag.reroute_flash > 0 else path_edges
            for i in range(ag.path_index, len(ag.path) - 1):
                target.add(self.edge_key(ag.path[i], ag.path[i + 1]))
        for u, v in path_edges:
            if self.G.has_edge(u, v) and not self.edge_is_blocked(u, v):
                x1, y1 = self.G.nodes[u]["pos"]
                x2, y2 = self.G.nodes[v]["pos"]
                pygame.draw.line(screen, PATH_COLOR, (x1, y1), (x2, y2), 2)
        for u, v in reroute_edges:
            if self.G.has_edge(u, v) and not self.edge_is_blocked(u, v):
                x1, y1 = self.G.nodes[u]["pos"]
                x2, y2 = self.G.nodes[v]["pos"]
                pygame.draw.line(screen, REROUTE_PATH_COLOR, (x1, y1), (x2, y2), 5)

        for n, data in self.G.nodes(data=True):
            x, y = data["pos"]
            if n in self.fire:
                pygame.draw.circle(screen, FIRE_COLOR, (int(x), int(y)), 11)
            else:
                smoke = self.smoke[n]
                color = NODE_COLOR
                if data["kind"] == "exit":
                    color = EXIT_COLOR
                elif data["kind"] == "stair":
                    color = STAIR_COLOR
                color = (
                    int(color[0] * (1 - smoke) + SMOKE_COLOR[0] * smoke),
                    int(color[1] * (1 - smoke) + SMOKE_COLOR[1] * smoke),
                    int(color[2] * (1 - smoke) + SMOKE_COLOR[2] * smoke),
                )
                pygame.draw.circle(screen, color, (int(x), int(y)), 8)
            if n.startswith("EX_") or "_S" in n:
                label = small_font.render(n, True, TEXT_COLOR)
                screen.blit(label, (x - label.get_width() // 2, y + 10))

        for ag in self.agents:
            if ag.evacuated:
                continue
            # [보완 5] 에이전트 타입 색상 구분
            if ag.agent_type == "elderly":
                color = ELDERLY_COLOR
            elif ag.agent_type == "child":
                color = (180, 220, 80)
            else:
                color = AGENT_COLOR
            radius = ag.radius
            if ag.fallback_mode:
                color = (200, 60, 200)
                radius += 1
            if ag.reroute_flash > 0:
                color = REROUTE_PATH_COLOR
                radius += 1
            pygame.draw.circle(screen, color, (int(ag.x), int(ag.y)), radius)

        total = len(self.agents)
        evacuated = sum(a.evacuated for a in self.agents)
        blocked_corridors = len(self.blocked_edges_dynamic)
        hot_nodes = sum(1 for t in self.temperature.values() if t >= 60.0)
        flashing = sum(1 for a in self.agents if (not a.evacuated) and a.reroute_flash > 0)
        fallback_now = sum(1 for a in self.agents if (not a.evacuated) and a.fallback_mode)
        stranded = total - evacuated
        recent_events = [label for _, label in self.event_log[-2:]]

        # [보완 7] 통계 화면 보강
        evac_times = sorted([a.evacuation_time for a in self.agents if a.evacuation_time is not None])
        p50 = evac_times[len(evac_times)//2] if evac_times else 0
        p90_idx = min(len(evac_times)-1, int(0.90*(len(evac_times)-1)))
        p90 = evac_times[p90_idx] if evac_times else 0

        info_lines = [
            self.scenario_name,
            f"Time: {self.time_s:5.1f}s | Floors: {len(FLOORS)}",
            f"Evacuated: {evacuated}/{total} | Stranded: {stranded}",
            f"P50: {p50:.1f}s | P90: {p90:.1f}s",
            f"Fire nodes: {len(self.fire)} | Blocked corridors: {blocked_corridors}",
            f"Hot nodes(>=60C): {hot_nodes}",
            f"Rerouting now: {flashing} | Total reroutes: {self.reroute_events}",
            f"Fallback now: {fallback_now}",
            "Keys: 1-3 scenario | SPACE pause | R reset | E export",
            *[f"Event: {line}" for line in recent_events]
        ]

        x0, y0 = 40, 8
        for i, line in enumerate(info_lines):
            surf = font.render(line, True, TEXT_COLOR)
            screen.blit(surf, (x0, y0 + i * 18))

        legend = pygame.Rect(WIDTH - 340, 8, 310, 270)
        pygame.draw.rect(screen, LEGEND_BG, legend, border_radius=8)
        pygame.draw.rect(screen, (180, 188, 198), legend, 1, border_radius=8)
        items = [
            (FIRE_COLOR, "Fire"),
            (SMOKE_COLOR, "Smoke / hazardous zone"),
            (HEAT_COLOR, "Heat rise zone"),
            (GUIDE_GO_COLOR, "Light guide: go"),
            (GUIDE_CAUTION_COLOR, "Light guide: caution"),
            (GUIDE_STOP_COLOR, "Light guide: stop"),
            (PATH_COLOR, "Recommended path"),
            (REROUTE_PATH_COLOR, "Recently rerouted"),
            ((200, 60, 200), "Fallback / deadlock escape"),
            (FIRE_DOOR_OPEN_COLOR, "Fire door open"),
            (FIRE_DOOR_CLOSED_COLOR, "Fire door closed"),
            (DANGER_COLOR, "Blocked corridor"),
            (ELDERLY_COLOR, "Elderly agent"),
            ((180, 220, 80), "Child agent"),
        ]
        for i, (c, label) in enumerate(items):
            y = legend.y + 15 + i * 18
            pygame.draw.circle(screen, c, (legend.x + 14, y + 6), 5)
            screen.blit(small_font.render(label, True, TEXT_COLOR), (legend.x + 28, y))

        if self.finished_reason:
            surf = big_font.render(f"FINISHED: {self.finished_reason}", True, (160, 50, 40))
            screen.blit(surf, (WIDTH - 440, 210))


def main():
    sim = EvacuationSim()
    paused = False
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim.setup_scenario(sim.scenario_id)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    sim.setup_scenario(int(event.unicode))
                elif event.key == pygame.K_e:
                    sim.export_results()

        if not paused:
            sim.step(dt)

        sim.draw(screen)

        if paused:
            surf = big_font.render("PAUSED", True, (150, 40, 40))
            screen.blit(surf, (WIDTH - 140, 15))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

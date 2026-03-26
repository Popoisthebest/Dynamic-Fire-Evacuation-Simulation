"""
Sensor-driven Dynamic Fire Evacuation Simulator (Headless, no external deps)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Tuple


AGENT_TYPES = {
    "normal": {"speed": 1.0, "reaction_delay": 0.2, "weight": 0.70},
    "elderly": {"speed": 0.65, "reaction_delay": 1.2, "weight": 0.18},
    "child": {"speed": 0.85, "reaction_delay": 0.5, "weight": 0.12},
}


@dataclass
class Agent:
    aid: int
    node: str
    agent_type: str
    speed: float
    reaction_delay: float
    path: List[str] = field(default_factory=list)
    path_idx: int = 0
    evacuated: bool = False
    reached_exit: str = ""
    evacuation_time: float | None = None
    reroute_count: int = 0
    stalled_time: float = 0.0
    smoke_exposure: float = 0.0
    distance_travelled: float = 0.0
    stranded_reason: str = ""


class GraphLite:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.adj: Dict[str, Dict[str, float]] = {}

    def add_node(self, node: str, **attrs):
        self.nodes[node] = attrs
        self.adj.setdefault(node, {})

    def add_edge(self, a: str, b: str, length: float):
        self.adj.setdefault(a, {})[b] = length
        self.adj.setdefault(b, {})[a] = length

    def neighbors(self, node: str):
        return self.adj.get(node, {}).items()

    def has_node(self, node: str) -> bool:
        return node in self.nodes


class SensorDrivenEvacuationSimulator:
    def __init__(self, scenario: str = "baseline", seed: int = 7):
        self.rng = random.Random(seed)
        self.scenario_name = scenario
        self.time_s = 0.0
        self.dt = 0.5
        self.max_time = 180.0

        self.risk_weights = {"smoke": 0.50, "temperature": 0.30, "congestion": 0.20}

        self.graph = self._build_graph()
        self.exits = [n for n, d in self.graph.nodes.items() if d["kind"] == "exit"]

        self.smoke = {n: 0.0 for n in self.graph.nodes}
        self.temperature = {n: 22.0 for n in self.graph.nodes}
        self.node_congestion = {n: 0 for n in self.graph.nodes}
        self.guidance = {n: "go" for n in self.graph.nodes}

        self.fire_nodes: set[str] = set()
        self.blocked_nodes: set[str] = set()
        self.event_log: list[dict] = []
        self.events: list[dict] = []

        self.agents: list[Agent] = []
        self._setup_scenario(scenario)
        self._spawn_agents()
        self._update_all_sensors()
        self._replan_all_agents(force=True)

    def _build_graph(self) -> GraphLite:
        g = GraphLite()
        floors = [3, 2, 1]
        cols = list(range(1, 13))

        for floor in floors:
            for col in cols:
                node = f"F{floor}_C{col}"
                g.add_node(node, floor=floor, col=col, kind="corridor")

        for floor in floors:
            for col in cols[:-1]:
                g.add_edge(f"F{floor}_C{col}", f"F{floor}_C{col+1}", length=6.0)

        for upper, lower in [(3, 2), (2, 1)]:
            for s in (2, 6, 10):
                g.add_edge(f"F{upper}_C{s}", f"F{lower}_C{s}", length=8.0)

        g.add_node("EX_L", floor=1, col=0, kind="exit")
        g.add_node("EX_C", floor=1, col=6, kind="exit")
        g.add_node("EX_R", floor=1, col=13, kind="exit")
        g.add_edge("F1_C1", "EX_L", length=4.0)
        g.add_edge("F1_C6", "EX_C", length=4.0)
        g.add_edge("F1_C12", "EX_R", length=4.0)
        return g

    def _setup_scenario(self, scenario: str) -> None:
        scenarios = {
            "baseline": {
                "fire": ["F2_C7"],
                "blocked": [],
                "events": [
                    {"time": 15.0, "type": "block_node", "node": "EX_C", "label": "Center exit closed"},
                    {"time": 25.0, "type": "ignite", "node": "F3_C6", "label": "Secondary fire at 3F core"},
                ],
            },
            "left_fire": {
                "fire": ["F3_C3"],
                "blocked": ["F1_C2"],
                "events": [{"time": 20.0, "type": "block_node", "node": "EX_L", "label": "Left exit blocked"}],
            },
            "right_fire": {
                "fire": ["F3_C10"],
                "blocked": [],
                "events": [{"time": 12.0, "type": "block_node", "node": "EX_R", "label": "Right exit blocked"}],
            },
        }
        conf = scenarios[scenario]
        self.fire_nodes = set(conf["fire"])
        self.blocked_nodes = set(conf["blocked"])
        self.events = conf["events"]

    def _spawn_agents(self) -> None:
        candidates = [
            n for n, d in self.graph.nodes.items() if d["kind"] == "corridor" and d["floor"] in (2, 3) and n not in self.fire_nodes
        ]

        def pick_type() -> str:
            r = self.rng.random()
            c = 0.0
            for t, cfg in AGENT_TYPES.items():
                c += cfg["weight"]
                if r <= c:
                    return t
            return "normal"

        aid = 0
        for node in candidates:
            cnt = 3 if node.startswith("F3") else 2
            for _ in range(cnt):
                at = pick_type()
                cfg = AGENT_TYPES[at]
                self.agents.append(
                    Agent(
                        aid=aid,
                        node=node,
                        agent_type=at,
                        speed=cfg["speed"] * self.rng.uniform(0.9, 1.1),
                        reaction_delay=cfg["reaction_delay"],
                    )
                )
                aid += 1

    def _run_events(self):
        for ev in self.events:
            if ev.get("done") or self.time_s < ev["time"]:
                continue
            if ev["type"] == "block_node":
                self.blocked_nodes.add(ev["node"])
            elif ev["type"] == "ignite":
                self.fire_nodes.add(ev["node"])
            ev["done"] = True
            self.event_log.append({"time": round(self.time_s, 2), "label": ev["label"]})

    def _spread_fire(self):
        if self.time_s == 0 or int(self.time_s) % 20 != 0:
            return
        to_add = []
        for f in list(self.fire_nodes):
            neighbors = [n for n, _ in self.graph.neighbors(f) if n not in self.fire_nodes and self.graph.nodes[n]["kind"] != "exit"]
            if neighbors:
                to_add.append(self.rng.choice(neighbors))
        for n in to_add[:2]:
            self.fire_nodes.add(n)

    def _dijkstra_length(self, source: str, blocked: set[str]) -> Dict[str, float]:
        dist = {source: 0.0}
        pq = [(0.0, source)]
        while pq:
            d, u = heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in self.graph.neighbors(u):
                if v in blocked:
                    continue
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heappush(pq, (nd, v))
        return dist

    def _update_sensor_fields(self):
        blocked = set(self.blocked_nodes)
        for n in self.graph.nodes:
            if n in blocked:
                self.smoke[n] = 0.0
                self.temperature[n] = 22.0
                continue
            if n in self.fire_nodes:
                self.smoke[n] = 1.0
                self.temperature[n] = 260.0
                continue

            min_dist = math.inf
            for f in self.fire_nodes:
                if f in blocked:
                    continue
                dmap = self._dijkstra_length(f, blocked)
                min_dist = min(min_dist, dmap.get(n, math.inf))

            if math.isinf(min_dist):
                self.smoke[n] = 0.0
                self.temperature[n] = 22.0
            else:
                front = max(0.0, self.time_s * 1.4 - min_dist)
                self.smoke[n] = max(0.0, min(1.0, front / 20.0))
                temp_gain = max(0.0, (35.0 - min_dist) / 35.0)
                self.temperature[n] = 22.0 + 85.0 * temp_gain

    def _update_congestion(self):
        c = {n: 0 for n in self.graph.nodes}
        for a in self.agents:
            if not a.evacuated:
                c[a.node] += 1
        self.node_congestion = c

    def _node_risk(self, node: str) -> float:
        if node in self.blocked_nodes or node in self.fire_nodes:
            return 1.0
        smoke = self.smoke[node]
        temp_norm = min(1.0, max(0.0, (self.temperature[node] - 22.0) / 90.0))
        cong_norm = min(1.0, self.node_congestion[node] / 8.0)
        return min(1.0, smoke * 0.5 + temp_norm * 0.3 + cong_norm * 0.2)

    def _update_guidance(self):
        for n in self.graph.nodes:
            r = self._node_risk(n)
            self.guidance[n] = "stop" if r >= 0.72 else "caution" if r >= 0.40 else "go"

    def _heuristic(self, a: str, b: str) -> float:
        na, nb = self.graph.nodes[a], self.graph.nodes[b]
        return abs(na.get("floor", 1) - nb.get("floor", 1)) * 4.0 + abs(na.get("col", 0) - nb.get("col", 0)) * 1.5

    def _transition_cost(self, u: str, v: str, length: float) -> float:
        if u in self.blocked_nodes or v in self.blocked_nodes or u in self.fire_nodes or v in self.fire_nodes:
            return math.inf
        g_pen = 400.0 if self.guidance.get(v) == "stop" else 30.0 if self.guidance.get(v) == "caution" else 0.0
        return length + (self._node_risk(u) + self._node_risk(v)) * 120.0 + g_pen

    def _a_star_between(self, source: str, goal: str) -> Tuple[List[str], float]:
        if source in self.blocked_nodes or source in self.fire_nodes:
            return [], math.inf
        pq = [(0.0, source)]
        came = {}
        gscore = {source: 0.0}

        while pq:
            _, u = heappop(pq)
            if u == goal:
                return self._reconstruct(came, u), gscore[u]
            for v, length in self.graph.neighbors(u):
                step = self._transition_cost(u, v, length)
                if math.isinf(step):
                    continue
                ng = gscore[u] + step
                if ng < gscore.get(v, math.inf):
                    came[v] = u
                    gscore[v] = ng
                    heappush(pq, (ng + self._heuristic(v, goal), v))

        return [], math.inf

    @staticmethod
    def _reconstruct(came: Dict[str, str], cur: str) -> List[str]:
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        return path

    def _a_star_safest_path(self, source: str) -> List[str]:
        best, best_cost = [], math.inf
        for ex in self.exits:
            if ex in self.blocked_nodes:
                continue
            p, c = self._a_star_between(source, ex)
            if p and c < best_cost:
                best, best_cost = p, c
        return best

    def _replan_all_agents(self, force=False):
        for a in self.agents:
            if a.evacuated or (a.reaction_delay > 0 and not force):
                continue
            newp = self._a_star_safest_path(a.node)
            old = a.path[a.path_idx:] if a.path else []
            if not newp:
                a.path, a.path_idx = [], 0
            elif force or old != newp:
                if old:
                    a.reroute_count += 1
                a.path, a.path_idx = newp, 0

    def _move_agents(self):
        for a in self.agents:
            if a.evacuated:
                continue
            if a.reaction_delay > 0:
                a.reaction_delay = max(0.0, a.reaction_delay - self.dt)
                a.stalled_time += self.dt
                continue

            a.smoke_exposure += self.smoke[a.node] * self.dt

            if a.node in self.exits and a.node not in self.blocked_nodes:
                a.evacuated = True
                a.reached_exit = a.node
                a.evacuation_time = self.time_s
                continue

            if not a.path or a.path_idx + 1 >= len(a.path):
                a.stalled_time += self.dt
                continue

            nxt = a.path[a.path_idx + 1]
            length = self.graph.adj[a.node][nxt]
            risk = self._node_risk(nxt)
            # 이산 시간 시뮬레이션: 이동 확률 기반으로 에지 통과 처리
            base_prob = (a.speed * self.dt) / max(1e-6, length)
            move_prob = min(0.95, max(0.08, base_prob * (2.8 - 1.5 * risk)))
            if self.rng.random() < move_prob:
                a.distance_travelled += length
                a.node = nxt
                a.path_idx += 1
                a.stalled_time = max(0.0, a.stalled_time - 0.25)
            else:
                a.stalled_time += self.dt

    def _mark_stranded(self):
        for a in self.agents:
            if a.evacuated:
                continue
            if not a.path:
                a.stranded_reason = "no_safe_path"
            elif a.stalled_time > 20:
                a.stranded_reason = "severe_congestion"
            else:
                a.stranded_reason = "time_limit"

    def _update_all_sensors(self):
        self._update_congestion()
        self._update_sensor_fields()
        self._update_guidance()

    def step(self):
        self.time_s += self.dt
        self._run_events()
        self._spread_fire()
        self._update_all_sensors()
        if int(self.time_s * 2) != int((self.time_s - self.dt) * 2):
            self._replan_all_agents()
        self._move_agents()

    def run(self, max_steps=400):
        for _ in range(max_steps):
            if self.time_s >= self.max_time or all(a.evacuated for a in self.agents):
                break
            self.step()
        if not all(a.evacuated for a in self.agents):
            self._mark_stranded()

    def summary(self) -> dict:
        evac = sorted([a.evacuation_time for a in self.agents if a.evacuation_time is not None])

        def pct(vals, p):
            if not vals:
                return None
            i = min(len(vals) - 1, int((len(vals) - 1) * p))
            return vals[i]

        exit_usage, stranded = {}, {}
        for a in self.agents:
            if a.reached_exit:
                exit_usage[a.reached_exit] = exit_usage.get(a.reached_exit, 0) + 1
            if not a.evacuated and a.stranded_reason:
                stranded[a.stranded_reason] = stranded.get(a.stranded_reason, 0) + 1

        gcount = {"go": 0, "caution": 0, "stop": 0}
        for v in self.guidance.values():
            gcount[v] += 1

        return {
            "scenario": self.scenario_name,
            "sim_time_s": round(self.time_s, 2),
            "total_agents": len(self.agents),
            "evacuated_agents": sum(a.evacuated for a in self.agents),
            "stranded_agents": sum(not a.evacuated for a in self.agents),
            "evacuation_rate": round(sum(a.evacuated for a in self.agents) / max(1, len(self.agents)), 4),
            "avg_evac_time_s": round(sum(evac) / len(evac), 3) if evac else None,
            "p50_evac_time_s": round(pct(evac, 0.5), 3) if evac else None,
            "p90_evac_time_s": round(pct(evac, 0.9), 3) if evac else None,
            "avg_smoke_exposure": round(sum(a.smoke_exposure for a in self.agents) / max(1, len(self.agents)), 3),
            "avg_reroutes": round(sum(a.reroute_count for a in self.agents) / max(1, len(self.agents)), 3),
            "avg_temperature_c": round(sum(self.temperature.values()) / len(self.temperature), 3),
            "max_temperature_c": round(max(self.temperature.values()), 3),
            "guidance_counts": gcount,
            "exit_usage": exit_usage,
            "stranded_reasons": stranded,
            "events": self.event_log,
        }

    def export(self, out_dir=".", prefix=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = prefix or f"evac_{self.scenario_name}_{stamp}"

        sp = out / f"{base}_summary.json"
        with sp.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

        ap = out / f"{base}_agents.csv"
        with ap.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "agent_id", "agent_type", "final_node", "evacuated", "reached_exit", "evacuation_time_s",
                "reroute_count", "stalled_time_s", "smoke_exposure", "distance_travelled", "stranded_reason",
            ])
            for a in self.agents:
                w.writerow([
                    a.aid, a.agent_type, a.node, a.evacuated, a.reached_exit,
                    "" if a.evacuation_time is None else round(a.evacuation_time, 3),
                    a.reroute_count, round(a.stalled_time, 3), round(a.smoke_exposure, 3),
                    round(a.distance_travelled, 3), a.stranded_reason,
                ])
        return sp, ap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="baseline", choices=["baseline", "left_fire", "right_fire"])
    p.add_argument("--steps", type=int, default=360)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--prefix", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sim = SensorDrivenEvacuationSimulator(args.scenario, args.seed)
    sim.run(args.steps)
    sp, ap = sim.export(args.out_dir, args.prefix)
    print(json.dumps(sim.summary(), ensure_ascii=False, indent=2))
    print(f"Saved summary: {sp}")
    print(f"Saved agents: {ap}")


if __name__ == "__main__":
    main()

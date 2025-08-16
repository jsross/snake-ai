#!/usr/bin/env python3
"""
Strategy Analytics Module

Provides detailed tracking and analysis of training strategy usage,
transitions, and performance metrics.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json


@dataclass
class StrategyTransition:
    """Record of a strategy transition"""
    episode: int
    from_strategy: str
    to_strategy: str
    reason: str = "performance_threshold"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_name: str
    episodes_active: int = 0
    total_score: float = 0.0
    total_reward: float = 0.0
    total_steps: int = 0
    survival_count: int = 0
    episodes: List[int] = field(default_factory=list)
    
    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.episodes_active)
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes_active)
    
    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(1, self.episodes_active)
    
    @property
    def survival_rate(self) -> float:
        return self.survival_count / max(1, self.episodes_active)


class StrategyTracker:
    """Tracks strategy usage and performance during training"""
    
    def __init__(self):
        self.current_strategy: Optional[str] = None
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.transitions: List[StrategyTransition] = []
        self.episode_strategy_map: Dict[int, str] = {}
        
    def record_episode(self, episode: int, strategy_name: str, 
                      score: float, reward: float, steps: int, survived: bool):
        """Record performance data for an episode"""
        # Track strategy transition
        if self.current_strategy != strategy_name:
            if self.current_strategy is not None:
                self.transitions.append(StrategyTransition(
                    episode=episode,
                    from_strategy=self.current_strategy,
                    to_strategy=strategy_name
                ))
            self.current_strategy = strategy_name
        
        # Record episode strategy
        self.episode_strategy_map[episode] = strategy_name
        
        # Update strategy performance
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(strategy_name)
        
        perf = self.strategy_performance[strategy_name]
        perf.episodes_active += 1
        perf.total_score += score
        perf.total_reward += reward
        perf.total_steps += steps
        if survived:
            perf.survival_count += 1
        perf.episodes.append(episode)
    
    def get_strategy_summary(self) -> Dict[str, Dict]:
        """Get summary of all strategy performance"""
        summary = {}
        for name, perf in self.strategy_performance.items():
            summary[name] = {
                'episodes_active': perf.episodes_active,
                'avg_score': round(perf.avg_score, 2),
                'avg_reward': round(perf.avg_reward, 2),
                'avg_steps': round(perf.avg_steps, 1),
                'survival_rate': round(perf.survival_rate, 3),
                'episode_range': f"{min(perf.episodes)}-{max(perf.episodes)}" if perf.episodes else "None"
            }
        return summary
    
    def get_transition_analysis(self) -> Dict[str, int]:
        """Analyze strategy transitions"""
        transition_counts = Counter()
        for trans in self.transitions:
            key = f"{trans.from_strategy} → {trans.to_strategy}"
            transition_counts[key] += 1
        return dict(transition_counts)
    
    def get_strategy_timeline(self) -> List[Tuple[int, int, str]]:
        """Get timeline of strategy usage (start_episode, end_episode, strategy)"""
        if not self.transitions:
            return []
        
        timeline = []
        current_start = 1
        
        for trans in self.transitions:
            timeline.append((current_start, trans.episode - 1, trans.from_strategy))
            current_start = trans.episode
        
        # Add final strategy period
        if self.episode_strategy_map:
            last_episode = max(self.episode_strategy_map.keys())
            timeline.append((current_start, last_episode, self.current_strategy))
        
        return timeline
    
    def export_analytics(self, filepath: str):
        """Export strategy analytics to JSON file"""
        analytics = {
            'strategy_summary': self.get_strategy_summary(),
            'transitions': [
                {
                    'episode': t.episode,
                    'from': t.from_strategy,
                    'to': t.to_strategy,
                    'reason': t.reason
                }
                for t in self.transitions
            ],
            'transition_analysis': self.get_transition_analysis(),
            'strategy_timeline': [
                {
                    'start_episode': start,
                    'end_episode': end,
                    'strategy': strategy,
                    'duration': end - start + 1
                }
                for start, end, strategy in self.get_strategy_timeline()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(analytics, f, indent=2)
    
    def print_summary(self):
        """Print a formatted summary of strategy analytics"""
        print("\n" + "="*60)
        print("STRATEGY ANALYTICS SUMMARY")
        print("="*60)
        
        # Strategy Performance
        print("\n📊 STRATEGY PERFORMANCE:")
        summary = self.get_strategy_summary()
        for strategy, stats in summary.items():
            print(f"\n  {strategy.upper()}:")
            print(f"    Episodes Active: {stats['episodes_active']}")
            print(f"    Average Score: {stats['avg_score']}")
            print(f"    Average Reward: {stats['avg_reward']}")
            print(f"    Average Steps: {stats['avg_steps']}")
            print(f"    Survival Rate: {stats['survival_rate']:.1%}")
            print(f"    Episode Range: {stats['episode_range']}")
        
        # Transitions
        print(f"\n🔄 STRATEGY TRANSITIONS ({len(self.transitions)} total):")
        transition_analysis = self.get_transition_analysis()
        for transition, count in transition_analysis.items():
            print(f"    {transition}: {count} times")
        
        # Timeline
        print("\n📅 STRATEGY TIMELINE:")
        timeline = self.get_strategy_timeline()
        for start, end, strategy in timeline:
            duration = end - start + 1
            print(f"    Episodes {start:4d}-{end:4d}: {strategy:<15} ({duration:4d} episodes)")
        
        print("="*60)


# Global strategy tracker instance
strategy_tracker = StrategyTracker()

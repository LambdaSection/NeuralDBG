"""
SLO/SLA tracking for production monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class SLOType(Enum):
    """SLO types."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class SLO:
    """Service Level Objective."""
    
    name: str
    slo_type: SLOType
    target: float
    window_seconds: float
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'slo_type': self.slo_type.value,
            'target': self.target,
            'window_seconds': self.window_seconds,
            'description': self.description,
        }


@dataclass
class SLOMeasurement:
    """SLO measurement."""
    
    timestamp: float
    slo_name: str
    actual_value: float
    target_value: float
    is_meeting: bool
    breach_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'slo_name': self.slo_name,
            'actual_value': self.actual_value,
            'target_value': self.target_value,
            'is_meeting': self.is_meeting,
            'breach_duration': self.breach_duration,
        }


@dataclass
class SLAReport:
    """SLA report."""
    
    start_time: float
    end_time: float
    slo_name: str
    measurements: List[SLOMeasurement]
    compliance_rate: float
    total_breaches: int
    total_breach_duration: float
    max_breach_duration: float
    error_budget_remaining: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'slo_name': self.slo_name,
            'measurements': [m.to_dict() for m in self.measurements],
            'compliance_rate': self.compliance_rate,
            'total_breaches': self.total_breaches,
            'total_breach_duration': self.total_breach_duration,
            'max_breach_duration': self.max_breach_duration,
            'error_budget_remaining': self.error_budget_remaining,
        }


class SLOTracker:
    """Tracker for SLOs and SLAs."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None
    ):
        """
        Initialize SLO tracker.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store SLO data
        """
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/slo")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.slos: Dict[str, SLO] = {}
        self.measurements: Dict[str, List[SLOMeasurement]] = {}
        self.breach_start_times: Dict[str, Optional[float]] = {}
    
    def add_slo(self, slo: SLO):
        """Add an SLO to track."""
        self.slos[slo.name] = slo
        self.measurements[slo.name] = []
        self.breach_start_times[slo.name] = None
    
    def remove_slo(self, slo_name: str):
        """Remove an SLO."""
        if slo_name in self.slos:
            del self.slos[slo_name]
        if slo_name in self.measurements:
            del self.measurements[slo_name]
        if slo_name in self.breach_start_times:
            del self.breach_start_times[slo_name]
    
    def record_measurement(
        self,
        slo_name: str,
        actual_value: float,
        timestamp: Optional[float] = None
    ):
        """
        Record an SLO measurement.
        
        Parameters
        ----------
        slo_name : str
            Name of the SLO
        actual_value : float
            Actual measured value
        timestamp : float, optional
            Timestamp of measurement
        """
        if slo_name not in self.slos:
            raise ValueError(f"SLO not found: {slo_name}")
        
        slo = self.slos[slo_name]
        timestamp = timestamp or time.time()
        
        # Check if meeting SLO
        is_meeting = self._check_slo_met(slo, actual_value)
        
        # Calculate breach duration
        breach_duration = 0.0
        if not is_meeting:
            if self.breach_start_times[slo_name] is None:
                self.breach_start_times[slo_name] = timestamp
            else:
                breach_duration = timestamp - self.breach_start_times[slo_name]
        else:
            self.breach_start_times[slo_name] = None
        
        measurement = SLOMeasurement(
            timestamp=timestamp,
            slo_name=slo_name,
            actual_value=actual_value,
            target_value=slo.target,
            is_meeting=is_meeting,
            breach_duration=breach_duration
        )
        
        self.measurements[slo_name].append(measurement)
        self._save_measurement(measurement)
        
        # Clean old measurements outside window
        self._clean_old_measurements(slo_name, timestamp, slo.window_seconds)
    
    def _check_slo_met(self, slo: SLO, actual_value: float) -> bool:
        """Check if SLO is met."""
        if slo.slo_type == SLOType.AVAILABILITY:
            # Availability: actual >= target (e.g., 99.9%)
            return actual_value >= slo.target
        elif slo.slo_type == SLOType.LATENCY:
            # Latency: actual <= target (e.g., p95 < 100ms)
            return actual_value <= slo.target
        elif slo.slo_type == SLOType.ACCURACY:
            # Accuracy: actual >= target (e.g., accuracy >= 95%)
            return actual_value >= slo.target
        elif slo.slo_type == SLOType.ERROR_RATE:
            # Error rate: actual <= target (e.g., error rate < 1%)
            return actual_value <= slo.target
        elif slo.slo_type == SLOType.THROUGHPUT:
            # Throughput: actual >= target (e.g., requests/sec >= 100)
            return actual_value >= slo.target
        else:
            return True
    
    def _clean_old_measurements(self, slo_name: str, current_time: float, window_seconds: float):
        """Clean measurements outside the window."""
        cutoff_time = current_time - window_seconds
        self.measurements[slo_name] = [
            m for m in self.measurements[slo_name]
            if m.timestamp >= cutoff_time
        ]
    
    def get_slo_status(self, slo_name: str) -> Dict[str, Any]:
        """
        Get current SLO status.
        
        Parameters
        ----------
        slo_name : str
            Name of the SLO
            
        Returns
        -------
        dict
            SLO status
        """
        if slo_name not in self.slos:
            raise ValueError(f"SLO not found: {slo_name}")
        
        slo = self.slos[slo_name]
        measurements = self.measurements[slo_name]
        
        if not measurements:
            return {
                'status': 'no_data',
                'slo': slo.to_dict(),
                'message': 'No measurements available'
            }
        
        # Calculate compliance
        total_measurements = len(measurements)
        meeting_measurements = sum(m.is_meeting for m in measurements)
        compliance_rate = meeting_measurements / total_measurements
        
        # Calculate breaches
        breaches = [m for m in measurements if not m.is_meeting]
        total_breaches = len(breaches)
        total_breach_duration = sum(m.breach_duration for m in breaches)
        max_breach_duration = max((m.breach_duration for m in breaches), default=0.0)
        
        # Calculate error budget
        error_budget = 1.0 - slo.target
        error_budget_used = 1.0 - compliance_rate
        error_budget_remaining = max(0.0, (error_budget - error_budget_used) / error_budget)
        
        # Recent trend
        recent_window = min(10, len(measurements))
        recent_measurements = measurements[-recent_window:]
        recent_compliance = sum(m.is_meeting for m in recent_measurements) / recent_window
        
        return {
            'status': 'ok',
            'slo': slo.to_dict(),
            'compliance_rate': compliance_rate,
            'is_meeting': compliance_rate >= slo.target,
            'total_measurements': total_measurements,
            'total_breaches': total_breaches,
            'total_breach_duration': total_breach_duration,
            'max_breach_duration': max_breach_duration,
            'error_budget_remaining': error_budget_remaining,
            'recent_compliance': recent_compliance,
            'current_breach': self.breach_start_times[slo_name] is not None,
            'current_breach_duration': (
                time.time() - self.breach_start_times[slo_name]
                if self.breach_start_times[slo_name] is not None
                else 0.0
            )
        }
    
    def generate_sla_report(
        self,
        slo_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> SLAReport:
        """
        Generate SLA report for a time period.
        
        Parameters
        ----------
        slo_name : str
            Name of the SLO
        start_time : float, optional
            Start time for report
        end_time : float, optional
            End time for report
            
        Returns
        -------
        SLAReport
            SLA report
        """
        if slo_name not in self.slos:
            raise ValueError(f"SLO not found: {slo_name}")
        
        slo = self.slos[slo_name]
        end_time = end_time or time.time()
        start_time = start_time or (end_time - slo.window_seconds)
        
        # Filter measurements in time range
        measurements = [
            m for m in self.measurements[slo_name]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not measurements:
            return SLAReport(
                start_time=start_time,
                end_time=end_time,
                slo_name=slo_name,
                measurements=[],
                compliance_rate=1.0,
                total_breaches=0,
                total_breach_duration=0.0,
                max_breach_duration=0.0,
                error_budget_remaining=1.0
            )
        
        # Calculate metrics
        total_measurements = len(measurements)
        meeting_measurements = sum(m.is_meeting for m in measurements)
        compliance_rate = meeting_measurements / total_measurements
        
        breaches = [m for m in measurements if not m.is_meeting]
        total_breaches = len(breaches)
        total_breach_duration = sum(m.breach_duration for m in breaches)
        max_breach_duration = max((m.breach_duration for m in breaches), default=0.0)
        
        error_budget = 1.0 - slo.target
        error_budget_used = 1.0 - compliance_rate
        error_budget_remaining = max(0.0, (error_budget - error_budget_used) / error_budget)
        
        report = SLAReport(
            start_time=start_time,
            end_time=end_time,
            slo_name=slo_name,
            measurements=measurements,
            compliance_rate=compliance_rate,
            total_breaches=total_breaches,
            total_breach_duration=total_breach_duration,
            max_breach_duration=max_breach_duration,
            error_budget_remaining=error_budget_remaining
        )
        
        self._save_report(report)
        
        return report
    
    def get_all_slo_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all SLOs."""
        return {
            slo_name: self.get_slo_status(slo_name)
            for slo_name in self.slos
        }
    
    def _save_measurement(self, measurement: SLOMeasurement):
        """Save measurement to disk."""
        measurement_file = (
            self.storage_path / 
            f"measurement_{measurement.slo_name}_{int(measurement.timestamp)}.json"
        )
        with open(measurement_file, 'w') as f:
            json.dump(measurement.to_dict(), f, indent=2)
    
    def _save_report(self, report: SLAReport):
        """Save report to disk."""
        report_file = (
            self.storage_path / 
            f"report_{report.slo_name}_{int(report.start_time)}_{int(report.end_time)}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)


# Predefined SLOs
def create_availability_slo(target: float = 0.999, window_hours: int = 24) -> SLO:
    """Create availability SLO (e.g., 99.9% uptime)."""
    return SLO(
        name="model_availability",
        slo_type=SLOType.AVAILABILITY,
        target=target,
        window_seconds=window_hours * 3600,
        description=f"Model should be available {target*100:.1f}% of the time"
    )


def create_latency_slo(target_ms: float = 100.0, window_hours: int = 1) -> SLO:
    """Create latency SLO (e.g., p95 < 100ms)."""
    return SLO(
        name="prediction_latency_p95",
        slo_type=SLOType.LATENCY,
        target=target_ms / 1000.0,  # Convert to seconds
        window_seconds=window_hours * 3600,
        description=f"95th percentile latency should be below {target_ms}ms"
    )


def create_accuracy_slo(target: float = 0.95, window_hours: int = 24) -> SLO:
    """Create accuracy SLO (e.g., accuracy >= 95%)."""
    return SLO(
        name="model_accuracy",
        slo_type=SLOType.ACCURACY,
        target=target,
        window_seconds=window_hours * 3600,
        description=f"Model accuracy should be at least {target*100:.1f}%"
    )


def create_error_rate_slo(target: float = 0.01, window_hours: int = 1) -> SLO:
    """Create error rate SLO (e.g., error rate < 1%)."""
    return SLO(
        name="error_rate",
        slo_type=SLOType.ERROR_RATE,
        target=target,
        window_seconds=window_hours * 3600,
        description=f"Error rate should be below {target*100:.1f}%"
    )


def create_throughput_slo(target: float = 100.0, window_hours: int = 1) -> SLO:
    """Create throughput SLO (e.g., >= 100 requests/sec)."""
    return SLO(
        name="throughput",
        slo_type=SLOType.THROUGHPUT,
        target=target,
        window_seconds=window_hours * 3600,
        description=f"Throughput should be at least {target} requests/second"
    )

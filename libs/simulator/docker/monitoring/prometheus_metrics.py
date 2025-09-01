import re
import numpy as np
import requests
from abc import ABC, abstractmethod

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

class PrometheusMetrics(ABC):
    prometheus_url = PROMETHEUS_URL

    def get_metric(self, query, timestamp=None):
        if timestamp:
            query += f'&time={timestamp}'
        response = requests.get(f"{self.prometheus_url}?query={query}")
        if response.status_code == 200:
            return response.json()["data"]["result"]
        return None

    def fill_metric_array(self, data, stations, metric_key, regex, default_value):
        arr = np.full(stations, default_value, dtype=float)
        for entry in data:
            metric = entry.get("metric", {})
            field = metric.get(metric_key, "")
            match = re.match(regex, field)
            if match:
                index = int(match.group(1)) - 1
                arr[index] = float(entry.get("value", [None, "0"])[1])
        return arr

    @abstractmethod
    def get_arrival_rates(self, monitoring_period, stations, timestamp=None):
        pass

    @abstractmethod
    def get_queue_lengths(self, stations, timestamp=None):
        pass

    @abstractmethod
    def get_current_cores(self, stations, timestamp=None):
        pass
    
    @abstractmethod
    def get_current_think_times(self, monitoring_period, users, stations, timestamp=None):
        pass
    
    @abstractmethod
    def get_response_times(self, monitoring_period, stations, timestamp=None):
        pass
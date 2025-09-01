
from libs.simulator.docker.monitoring.prometheus_metrics import PrometheusMetrics


class ContainerMetrics(PrometheusMetrics):
    @staticmethod
    def ARRIVAL_RATE_QUERY(s):
        return f'rate(traefik_service_requests_total{{service=~"app.*@docker",code="200", method="GET"}}[{s}s])'
    @staticmethod
    def THINK_TIME_QUERY(s, users):
        return f'({users}*{s} - increase(traefik_service_request_duration_seconds_sum{{code="200"}}[{s}s]))/increase(traefik_service_requests_total{{code="200"}}[{s}s])'
    @staticmethod
    def RESPONSE_TIME_QUERY(s):
        return f'rate(traefik_service_request_duration_seconds_sum{{service=~"app.*@docker", code="200", method="GET"}}[{s}s]) / rate(traefik_service_request_duration_seconds_count{{service=~"app.*@docker", code="200", method="GET"}}[{s}s])'
    QUEUE_LENGTH_QUERY = 'traefik_container_open_connections{service=~"app.*"}'
    CURRENT_CORES_QUERY = 'container_spec_cpu_quota{name=~"app.*"} / container_spec_cpu_period{name=~"app.*"}'

    def get_arrival_rates(self, monitoring_period, stations, timestamp=None):
        query = self.ARRIVAL_RATE_QUERY(monitoring_period)
        data = self.get_metric(query, timestamp)
        return self.fill_metric_array(data, stations, "service", r'app(\d)@docker', 0.0)

    def get_queue_lengths(self, stations, timestamp=None):
        data = self.get_metric(self.QUEUE_LENGTH_QUERY, timestamp)
        return self.fill_metric_array(data, stations, "service", r'app(\d)@docker', 0.0)

    def get_current_cores(self, stations, timestamp=None):
        data = self.get_metric(self.CURRENT_CORES_QUERY, timestamp)
        return self.fill_metric_array(data, stations, "name", r'app(\d+)$', 1.0)
    
    def get_current_think_times(self, monitoring_period, users, stations, timestamp=None):
        query = self.THINK_TIME_QUERY(monitoring_period, users)
        data = self.get_metric(query, timestamp)
        return self.fill_metric_array(data, stations, "service", r'app(\d)@docker', 0.0)

    def get_response_times(self, monitoring_period, stations, timestamp=None):
        query = self.RESPONSE_TIME_QUERY(monitoring_period)
        data = self.get_metric(query, timestamp)
        return self.fill_metric_array(data, stations, "service", r'app(\d)@docker', 0.0)
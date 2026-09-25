"""Redis 连接管理层。

对外只暴露 RedisClient：业务代码不自己 `redis.Redis(...)`，
连接参数、超时、连接池和异常降级策略收敛在这一处。
"""

from frontend.local_rag.cache.redis_client import RedisClient

__all__ = ["RedisClient"]

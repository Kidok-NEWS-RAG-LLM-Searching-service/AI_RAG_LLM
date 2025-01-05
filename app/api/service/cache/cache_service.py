

class CacheService:

    def is_cache_hit(self, data):
        if len(data) > 0:
            return True
        return False


cache_service = CacheService()

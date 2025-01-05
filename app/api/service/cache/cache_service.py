

class CacheService:

    def is_cache_hit(self, query, data):
        query_embed = self.embeddings
        if len(data) > 0:
            return True
        return False


cache_service = CacheService()

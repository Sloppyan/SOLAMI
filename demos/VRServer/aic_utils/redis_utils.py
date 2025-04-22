import redis
import asyncio
from aic_utils.format_utils import format_data

class RedisConnection:
    def __init__(self, host, port, db, stream_key):
        self.r = redis.Redis(host=host, port=port, db=db)
        self.stream_key = stream_key
    
    def send_frame_data(self, frame_data):
        entries = {key: format_data(value) for key, value in frame_data.items() if key != "frame"}
        return self.r.xadd(self.stream_key, entries)
        
    def send_change_id(self, change_id):
        self.r.set("change_id", change_id)
        
    def clear_database(self):
        self.r.flushdb()
        
    def clear_ai_stream(self):
        self.r.xtrim(self.stream_key, maxlen=0)
        
    async def receive_frame_data(self, user_stream_key, last_id='0-0', count=300):
        entries = self.r.xread({user_stream_key: last_id}, count=count, block=5000)
        messages = []
        for stream, msgs in entries:
            for message_id, message in msgs:
                messages.append((message_id, {key.decode(): value.decode() for key, value in message.items()}))
                last_id = message_id
        return messages, last_id
    
    async def delete_processed_entries(self, stream_key, ids):
        result = await asyncio.to_thread(self.r.xdel, stream_key, *ids)
        print(f"Deleted {result} entries from {stream_key}")
        
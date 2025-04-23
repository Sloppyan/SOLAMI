import redis
import asyncio
from aic_utils.format_utils import format_data

class RedisConnection:
    """
    A class to handle Redis connection and operations for animation data streaming.
    
    This class provides methods to send, receive, and manage animation frame data
    using Redis streams, which enable efficient real-time communication between
    different components of the animation system.
    """
    
    def __init__(self, host, port, db, stream_key):
        """
        Initialize the Redis connection.
        
        Args:
            host (str): Redis server hostname or IP address
            port (int): Redis server port
            db (int): Redis database number
            stream_key (str): Default Redis stream key to use for operations
        """
        self.r = redis.Redis(host=host, port=port, db=db)
        self.stream_key = stream_key
    
    def send_frame_data(self, frame_data):
        """
        Send a frame of animation data to the Redis stream.
        
        This method formats each value in the frame data and adds it as an entry
        to the specified Redis stream, excluding the 'frame' key.
        
        Args:
            frame_data (dict): Dictionary containing animation data for a single frame
            
        Returns:
            str: The ID of the entry added to the stream
        """
        entries = {key: format_data(value) for key, value in frame_data.items() if key != "frame"}
        return self.r.xadd(self.stream_key, entries)
        
    def send_change_id(self, change_id):
        """
        Store a change identifier in Redis.
        
        This method is used to notify other components of a state change by
        setting a 'change_id' key in Redis.
        
        Args:
            change_id (str): Identifier for the change
        """
        self.r.set("change_id", change_id)
        
    def clear_database(self):
        """
        Clear all data in the current Redis database.
        
        This method removes all keys from the Redis database being used,
        effectively resetting the state.
        """
        self.r.flushdb()
        
    def clear_ai_stream(self):
        """
        Clear all entries from the AI stream.
        
        This method trims the stream to zero length, effectively removing
        all entries while keeping the stream key intact.
        """
        self.r.xtrim(self.stream_key, maxlen=0)
        
    async def receive_frame_data(self, user_stream_key, last_id='0-0', count=300):
        """
        Asynchronously receive animation frame data from a Redis stream.
        
        This method reads entries from the specified stream that have IDs greater
        than the provided last_id, up to the specified count. It blocks for up to
        5 seconds waiting for new data if none is immediately available.
        
        Args:
            user_stream_key (str): Redis stream key to read from
            last_id (str, optional): ID to start reading from. Defaults to '0-0'
                                    (beginning of the stream).
            count (int, optional): Maximum number of entries to read. Defaults to 300.
            
        Returns:
            tuple: (messages, last_id)
                  - messages: List of (message_id, message_data) tuples
                  - last_id: ID of the last message read, for use in subsequent calls
        """
        entries = self.r.xread({user_stream_key: last_id}, count=count, block=5000)
        messages = []
        for stream, msgs in entries:
            for message_id, message in msgs:
                messages.append((message_id, {key.decode(): value.decode() for key, value in message.items()}))
                last_id = message_id
        return messages, last_id
    
    async def delete_processed_entries(self, stream_key, ids):
        """
        Asynchronously delete entries from a Redis stream after they've been processed.
        
        This method removes entries with the specified IDs from the stream,
        helping to manage memory usage by cleaning up processed data.
        
        Args:
            stream_key (str): Redis stream key to delete entries from
            ids (list): List of entry IDs to delete
            
        Returns:
            int: Number of entries deleted
        """
        result = await asyncio.to_thread(self.r.xdel, stream_key, *ids)
        print(f"Deleted {result} entries from {stream_key}")
        
# chat_server.py

import socket
import threading
import json
import queue
from datetime import datetime
from typing import Dict, Set, Optional
import signal
import sys
from pathlib import Path
import pandas as pd

from logger_config import logger, log_exceptions
from config import config
from shared_types import AnalysisMetrics

class ClientHandler:
    """Handles individual client connections"""
    
    def __init__(self, socket: socket.socket, address: str, server: 'ChatServer'):
        """Initialize client handler"""
        self.socket = socket
        self.address = address
        self.server = server
        self.screen_name: Optional[str] = None
        self.connected = True
        self.message_queue = queue.Queue()
        self.last_activity = datetime.now()
        self.metrics_history: List[AnalysisMetrics] = []
        
        # Start message processing thread
        self.process_thread = threading.Thread(target=self.process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info(f"Client handler initialized for {address}")

    @log_exceptions()
    def handle(self):
        """Main client handling loop"""
        try:
            while self.connected:
                try:
                    # Receive data with timeout
                    self.socket.settimeout(config.network.timeout)
                    data = self.socket.recv(config.network.buffer_size)
                    
                    if not data:
                        break
                    
                    # Process received data
                    message = data.decode('utf-8')
                    self.message_queue.put(message)
                    self.last_activity = datetime.now()
                    
                except socket.timeout:
                    # Check if client is still alive
                    if (datetime.now() - self.last_activity).seconds > config.network.keepalive:
                        logger.warning(f"Client {self.address} timed out")
                        break
                    continue
                    
                except Exception as e:
                    logger.error(f"Error handling client {self.address}: {e}")
                    break
                    
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Fatal error handling client {self.address}: {e}")
            self.cleanup()

    @log_exceptions()
    def process_messages(self):
        """Process messages from queue"""
        while self.connected:
            try:
                message = self.message_queue.get(timeout=1.0)
                
                # Extract screen name from first message
                if not self.screen_name and ":" in message:
                    self.screen_name = message.split(":")[0].strip()
                    logger.info(f"Client {self.address} identified as {self.screen_name}")
                
                # Process message
                if message.startswith("METRICS:"):
                    # Handle metrics data
                    self.process_metrics(message[8:])
                else:
                    # Broadcast chat message
                    self.server.broadcast(message, exclude=self.socket)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message for {self.address}: {e}")

    @log_exceptions()
    def process_metrics(self, metrics_json: str):
        """Process metrics data from client"""
        try:
            metrics_dict = json.loads(metrics_json)
            metrics = AnalysisMetrics.from_dict(metrics_dict)
            self.metrics_history.append(metrics)
            
            # Update server statistics
            self.server.update_statistics(self.screen_name, metrics)
            
        except Exception as e:
            logger.error(f"Error processing metrics from {self.address}: {e}")

    def send(self, message: str):
        """Send message to client"""
        try:
            if self.connected:
                self.socket.send(message.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error sending to {self.address}: {e}")
            self.cleanup()

    def cleanup(self):
        """Clean up client resources"""
        try:
            self.connected = False
            self.socket.close()
            self.server.remove_client(self)
            logger.info(f"Client {self.address} disconnected")
            
        except Exception as e:
            logger.error(f"Error cleaning up client {self.address}: {e}")

class ChatServer:
    """Enhanced chat server with monitoring capabilities"""
    
    def __init__(self, host: str = config.network.server_ip, 
                 port: int = config.network.server_port):
        """Initialize chat server"""
        try:
            # Server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((host, port))
            
            # Client tracking
            self.clients: Set[ClientHandler] = set()
            self.client_lock = threading.Lock()
            
            # Statistics tracking
            self.statistics: Dict[str, pd.DataFrame] = {}
            self.stats_lock = threading.Lock()
            
            # Output directory
            self.output_dir = Path("server_output")
            self.output_dir.mkdir(exist_ok=True)
            
            # Server state
            self.running = False
            self.start_time = None
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
            
            logger.info(f"Chat server initialized on {host}:{port}")
            
        except Exception as e:
            logger.error(f"Error initializing server: {e}")
            raise

    def start(self):
        """Start the chat server"""
        try:
            self.running = True
            self.start_time = datetime.now()
            self.server_socket.listen(config.network.max_connections)
            
            logger.info("Chat server started")
            print(f"Server listening on {config.network.server_ip}:{config.network.server_port}")
            
            # Start statistics saving thread
            self.stats_thread = threading.Thread(target=self.save_statistics_periodically)
            self.stats_thread.daemon = True
            self.stats_thread.start()
            
            while self.running:
                try:
                    # Accept new connections
                    client_socket, address = self.server_socket.accept()
                    self.handle_new_client(client_socket, address)
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                    
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Fatal error in server: {e}")
            self.cleanup()

    @log_exceptions()
    def handle_new_client(self, client_socket: socket.socket, address: str):
        """Handle new client connection"""
        try:
            # Check max connections
            if len(self.clients) >= config.network.max_connections:
                logger.warning(f"Connection from {address} rejected: max connections reached")
                client_socket.close()
                return
            
            # Create client handler
            handler = ClientHandler(client_socket, address, self)
            
            # Add to clients set
            with self.client_lock:
                self.clients.add(handler)
            
            # Start handler thread
            client_thread = threading.Thread(target=handler.handle)
            client_thread.daemon = True
            client_thread.start()
            
            logger.info(f"New client connected from {address}")
            
        except Exception as e:
            logger.error(f"Error handling new client {address}: {e}")
            client_socket.close()

    @log_exceptions()
    def broadcast(self, message: str, exclude: Optional[socket.socket] = None):
        """Broadcast message to all clients"""
        with self.client_lock:
            for client in self.clients:
                if client.socket != exclude:
                    client.send(message)

    @log_exceptions()
    def update_statistics(self, screen_name: str, metrics: AnalysisMetrics):
        """Update server statistics"""
        try:
            with self.stats_lock:
                if screen_name not in self.statistics:
                    self.statistics[screen_name] = pd.DataFrame()
                
                # Add new metrics
                metrics_dict = metrics.to_dict()
                metrics_dict['timestamp'] = datetime.now()
                
                self.statistics[screen_name] = self.statistics[screen_name].append(
                    metrics_dict,
                    ignore_index=True
                )
                
        except Exception as e:
            logger.error(f"Error updating statistics for {screen_name}: {e}")

    def save_statistics_periodically(self):
        """Periodically save statistics to disk"""
        while self.running:
            try:
                # Save every 5 minutes
                import time
                time.sleep(300)
                
                self.save_statistics()
                
            except Exception as e:
                logger.error(f"Error in periodic statistics saving: {e}")

    @log_exceptions()
    def save_statistics(self):
        """Save current statistics to disk"""
        try:
            with self.stats_lock:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for screen_name, df in self.statistics.items():
                    if not df.empty:
                        filename = self.output_dir / f"stats_{screen_name}_{timestamp}.xlsx"
                        df.to_excel(filename, index=False)
                
                logger.info("Statistics saved to disk")
                
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")

    def remove_client(self, client: ClientHandler):
        """Remove client from server"""
        try:
            with self.client_lock:
                self.clients.discard(client)
                
            # Notify other clients
            if client.screen_name:
                self.broadcast(f"System: {client.screen_name} has left the chat")
                
        except Exception as e:
            logger.error(f"Error removing client: {e}")

    def cleanup(self):
        """Clean up server resources"""
        try:
            # Save final statistics
            self.save_statistics()
            
            # Close all client connections
            with self.client_lock:
                for client in self.clients:
                    client.cleanup()
                self.clients.clear()
            
            # Close server socket
            self.server_socket.close()
            
            logger.info("Server cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during server cleanup: {e}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def get_server_stats(self) -> Dict:
        """Get current server statistics"""
        try:
            stats = {
                'uptime': str(datetime.now() - self.start_time),
                'total_clients': len(self.clients),
                'total_messages': sum(
                    len(df) for df in self.statistics.values()
                ),
                'active_users': len([
                    client for client in self.clients
                    if client.screen_name is not None
                ])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting server stats: {e}")
            return {}

def main():
    """Main entry point"""
    try:
        server = ChatServer()
        server.start()
        
    except Exception as e:
        logger.error(f"Fatal error in server main: {e}")
        raise

if __name__ == "__main__":
    main()
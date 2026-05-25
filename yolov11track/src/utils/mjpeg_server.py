"""
MJPEG流服务器 - 用于远程查看检测结果
"""

import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import time


class MJPEGServer:
    """MJPEG流服务器"""
    
    def __init__(self, port=8080):
        """
        初始化服务器
        
        Args:
            port: 服务器端口
        """
        self.port = port
        self.frame = None
        self.frame_lock = threading.Lock()
        self.server = None
        self.server_thread = None
        self.running = False
        
    def update_frame(self, frame):
        """
        更新当前帧
        
        Args:
            frame: 图像帧
        """
        with self.frame_lock:
            self.frame = frame.copy()
    
    def get_frame(self):
        """获取当前帧"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def start(self):
        """启动服务器"""
        if self.running:
            print("MJPEG服务器已在运行")
            return False
        
        self.running = True
        
        # 创建服务器
        server_instance = self
        
        class StreamingHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    # 返回HTML页面
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = f"""
                    <html>
                    <head>
                        <title>Detection Pipeline Stream</title>
                        <style>
                            body {{
                                margin: 0;
                                padding: 20px;
                                background-color: #1e1e1e;
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                font-family: Arial, sans-serif;
                            }}
                            h1 {{
                                color: #00ff00;
                                margin-bottom: 20px;
                            }}
                            img {{
                                max-width: 100%;
                                border: 2px solid #00ff00;
                                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
                            }}
                            .info {{
                                color: #ffffff;
                                margin-top: 20px;
                                text-align: center;
                            }}
                        </style>
                    </head>
                    <body>
                        <h1>🎥 Detection Pipeline Live Stream</h1>
                        <img src="/stream" />
                        <div class="info">
                            <p>实时检测流 - 端口 {server_instance.port}</p>
                            <p>在浏览器中访问: http://&lt;开发板IP&gt;:{server_instance.port}</p>
                        </div>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                
                elif self.path == '/stream':
                    # 返回MJPEG流
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    
                    try:
                        while server_instance.running:
                            frame = server_instance.get_frame()
                            if frame is not None:
                                # 编码为JPEG
                                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                
                                # 发送帧
                                self.wfile.write(b'--frame\r\n')
                                self.send_header('Content-type', 'image/jpeg')
                                self.send_header('Content-length', len(jpeg))
                                self.end_headers()
                                self.wfile.write(jpeg.tobytes())
                                self.wfile.write(b'\r\n')
                            
                            time.sleep(0.033)  # ~30 FPS
                    
                    except Exception as e:
                        print(f"流传输错误: {e}")
                
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                # 禁用默认日志
                pass
        
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            pass
        
        try:
            self.server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamingHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            print(f"✓ MJPEG服务器启动成功")
            print(f"  访问地址: http://<开发板IP>:{self.port}")
            print(f"  在浏览器中打开即可查看实时视频流")
            return True
        
        except Exception as e:
            print(f"✗ MJPEG服务器启动失败: {e}")
            self.running = False
            return False
    
    def stop(self):
        """停止服务器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        
        print("MJPEG服务器已停止")


# 测试代码
if __name__ == '__main__':
    import numpy as np
    
    print("=" * 60)
    print("MJPEG服务器测试")
    print("=" * 60)
    
    # 创建服务器
    server = MJPEGServer(port=8080)
    
    # 启动服务器
    if server.start():
        print("\n生成测试视频流...")
        print("在浏览器中访问: http://localhost:8080")
        print("按 Ctrl+C 停止")
        
        try:
            # 生成测试帧
            frame_count = 0
            while True:
                # 创建测试帧
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 绘制文本
                text = f"Frame: {frame_count}"
                cv2.putText(frame, text, (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                # 更新帧
                server.update_frame(frame)
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\n\n收到停止信号...")
        
        # 停止服务器
        server.stop()
    
    print("\n测试完成")

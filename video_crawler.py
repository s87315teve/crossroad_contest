import pandas as pd
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import multiprocessing
import time
import os
from datetime import datetime
import signal
import sys

class CCTVRecorder:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.page_urls = self.load_urls_from_csv()
        self.locations_data = self.extract_locations_and_video_urls()
        self.running = True
        
    def signal_handler(self, signum, frame):
        """處理中斷信號"""
        print(f"\n收到停止信號，正在結束錄製...")
        self.running = False
        
    def load_urls_from_csv(self):
        """從CSV檔案讀取URL列表"""
        try:
            df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            # 假設CSV只有一個欄位叫做 'URL'，你可以根據實際欄位名稱調整
            if 'URL' in df.columns:
                return df['URL'].tolist()
            else:
                # 如果欄位名稱不同，取第一個欄位
                return df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"讀取CSV檔案時發生錯誤: {e}")
            return []
    
    def extract_location_from_page(self, page_url):
        """從網頁中提取地點名稱（從h1標籤中）"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 尋找h1標籤中的地點名稱
            h1_tag = soup.find('h1')
            if h1_tag:
                location_name = h1_tag.get_text(strip=True)
                # 移除「即時影像」等後綴，只保留地點名稱
                location_name = re.sub(r'即時影像$', '', location_name).strip()
                return location_name
            else:
                print(f"無法在 {page_url} 中找到h1標籤")
                return None
                
        except Exception as e:
            print(f"提取地點名稱時發生錯誤 ({page_url}): {e}")
            return None
    
    def extract_video_url_from_page(self, page_url):
        """從網頁中提取實際的影像URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 尋找包含影像的img標籤
            img_tags = soup.find_all('img', class_='cctv-image')
            if not img_tags:
                img_tags = soup.find_all('img')
            
            for img in img_tags:
                src = img.get('src', '')
                # 尋找包含串流URL的模式
                if 'zms' in src or 'cgi-bin' in src or any(x in src for x in ['nvr', 'cam', 'stream']):
                    # 解碼HTML實體
                    video_url = src.replace('&amp;', '&')
                    return video_url
            
            # 如果在img標籤中找不到，嘗試在JavaScript中尋找
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # 尋找可能的URL模式
                    url_patterns = re.findall(r'https?://[^\s"\']+(?:zms|stream|cgi-bin)[^\s"\']*', script.string)
                    if url_patterns:
                        return url_patterns[0].replace('&amp;', '&')
            
            print(f"無法從 {page_url} 提取影像URL")
            return None
            
        except Exception as e:
            print(f"提取影像URL時發生錯誤 ({page_url}): {e}")
            return None
    
    def extract_locations_and_video_urls(self):
        """提取所有地點名稱和對應的影像URL"""
        locations_data = {}
        
        for page_url in self.page_urls:
            print(f"正在處理: {page_url}")
            
            # 提取地點名稱
            location_name = self.extract_location_from_page(page_url)
            if not location_name:
                # 如果無法提取地點名稱，使用URL的一部分作為備用名稱
                location_name = page_url.split('/')[-1] if page_url.split('/')[-1] else page_url.split('/')[-2]
                print(f"使用備用地點名稱: {location_name}")
            
            # 提取影像URL
            video_url = self.extract_video_url_from_page(page_url)
            
            if video_url:
                locations_data[location_name] = video_url
                print(f"✓ {location_name}: {video_url}")
            else:
                print(f"✗ {location_name}: 無法提取影像URL")
        
        return locations_data
    
    def record_and_display_stream(self, location, video_url):
        """錄製和顯示單一串流的函數"""
        print(f"開始處理 {location} 的串流...")
        
        # 確保儲存目錄存在
        # 清理檔案名稱中的特殊字符
        safe_location = re.sub(r'[<>:"/\\|?*]', '_', location)
        save_dir = f"recordings/{safe_location}"
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                print(f"無法開啟 {location} 的影像串流")
                return
            
            # 取得影像屬性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # 如果無法取得FPS，設定預設值
                fps = 25.0
            
            print(f"{location} - 解析度: {width}x{height}, FPS: {fps}")
            
            # 設定影片編碼器 (H.264) - 嘗試多種編碼器
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'H264'),  # 嘗試H264
                cv2.VideoWriter_fourcc(*'XVID'),  # 嘗試XVID
                cv2.VideoWriter_fourcc(*'mp4v'),  # 嘗試mp4v
                cv2.VideoWriter_fourcc(*'MJPG')   # 嘗試MJPG
            ]
            
            # 修正FPS值，確保不會太高或太低
            if fps <= 0 or fps > 60:
                fps = 25.0
            elif fps < 10:
                fps = 25.0
            
            print(f"{location} - 使用FPS: {fps}")
            
            # 用於控制錄製時間的變數
            segment_duration = 30  # 30秒一段
            start_time = time.time()
            segment_count = 0
            frame_count = 0  # 添加幀計數器
            
            # 嘗試不同的編碼器直到找到可用的
            out = None
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for fourcc in fourcc_options:
                video_filename = f"{save_dir}/{safe_location}_{timestamp}_seg{segment_count:03d}.mp4"
                temp_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                if temp_out.isOpened():
                    out = temp_out
                    fourcc_name = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                    print(f"{location} - 使用編碼器: {fourcc_name}")
                    break
                else:
                    temp_out.release()
            
            if out is None:
                print(f"{location} - 無法建立影片寫入器")
                cap.release()
                return
            
            print(f"開始錄製 {location}, 檔案: {video_filename}")
            
            # 為每個攝影機創建唯一的視窗名稱，避免視窗名稱衝突
            unique_window_name = f"{location}_{id(self)}"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"{location}: 無法讀取畫面")
                    break
                
                # 寫入影片檔案
                if out and out.isOpened():
                    out.write(frame)
                    frame_count += 1
                
                # 顯示畫面 - 使用唯一視窗名稱
                window_display_name = f"{location} - 按 'q' 結束, 's' 手動存檔 (已錄製{frame_count}幀)"
                cv2.namedWindow(unique_window_name, cv2.WINDOW_NORMAL)
                cv2.setWindowTitle(unique_window_name, window_display_name)
                cv2.resizeWindow(unique_window_name, 640, 480)  # 調整顯示視窗大小
                cv2.imshow(unique_window_name, frame)
                
                # 檢查是否需要開始新的影片段落
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time >= segment_duration:
                    # 結束當前影片檔案
                    if out:
                        out.release()
                        print(f"完成錄製: {video_filename} (錄製了{frame_count}幀, 時長: {elapsed_time:.1f}秒)")
                    
                    # 開始新的影片檔案
                    segment_count += 1
                    frame_count = 0  # 重置幀計數器
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{save_dir}/{safe_location}_{timestamp}_seg{segment_count:03d}.mp4"
                    
                    # 使用相同的編碼器建立新的寫入器
                    for fourcc in fourcc_options:
                        temp_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                        if temp_out.isOpened():
                            out = temp_out
                            break
                        else:
                            temp_out.release()
                    
                    start_time = current_time
                    print(f"開始新的錄製: {video_filename}")
                
                # 檢查按鍵 - 增加延遲時間避免開啟太多視窗
                key = cv2.waitKey(30) & 0xFF  # 30ms延遲，約33FPS顯示
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 手動儲存當前畫面
                    frame_filename = f"{save_dir}/{safe_location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_frame.png"
                    cv2.imwrite(frame_filename, frame)
                    print(f"手動儲存畫面: {frame_filename}")
            
            # 清理資源
            if out:
                out.release()
                print(f"最終完成錄製: {video_filename} (總計{frame_count}幀)")
            cap.release()
            cv2.destroyWindow(unique_window_name)  # 只關閉這個特定的視窗
            print(f"{location} 錄製結束")
            
        except Exception as e:
            print(f"{location} 處理時發生錯誤: {e}")
            # 確保在錯誤情況下也清理視窗
            try:
                cv2.destroyWindow(unique_window_name)
            except:
                pass
    
    def start_recording(self):
        """開始多核心錄製"""
        if not self.locations_data:
            print("沒有可用的影像URL")
            return
        
        print(f"準備開始錄製 {len(self.locations_data)} 個串流...")
        
        # 建立錄製目錄
        os.makedirs("recordings", exist_ok=True)
        
        # 建立進程列表
        processes = []
        
        for location, video_url in self.locations_data.items():
            process = multiprocessing.Process(
                target=self.record_and_display_stream,
                args=(location, video_url)
            )
            processes.append(process)
            process.start()
            print(f"已啟動 {location} 的錄製進程")
        
        try:
            # 等待所有進程完成
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            print("\n正在停止所有錄製...")
            for process in processes:
                process.terminate()
                process.join()
            print("所有錄製已停止")

def main():
    # CSV檔案路徑
    csv_file = "video_urls.csv"
    
    # 檢查CSV檔案是否存在
    if not os.path.exists(csv_file):
        print(f"找不到CSV檔案: {csv_file}")
        print("請確保CSV檔案存在，格式為:")
        print("URL")
        print("https://tw.live/cam/?id=HSZ030")
        print("https://tw.live/cam/?id=另一個ID")
        return
    
    # 建立錄製器實例
    recorder = CCTVRecorder(csv_file)
    
    # 顯示找到的地點和URL
    print("\n找到的地點和影像URL:")
    for location, url in recorder.locations_data.items():
        print(f"- {location}: {url}")
    
    if recorder.locations_data:
        print(f"\n將開始錄製 {len(recorder.locations_data)} 個串流")
        print("每30秒自動切換新的影片檔案")
        print("按 'q' 可結束單一串流的錄製")
        print("按 's' 可手動儲存當前畫面")
        print("按 Ctrl+C 可停止所有錄製")
        
        input("\n按Enter開始錄製...")
        recorder.start_recording()
    else:
        print("沒有可用的影像串流")

if __name__ == "__main__":
    # 設定多進程啟動方法 (Windows相容性)
    multiprocessing.set_start_method('spawn', force=True)
    main()
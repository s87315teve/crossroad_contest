import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import json
import time
from pathlib import Path
import multiprocessing
import shutil
import glob

class YOLOVideoAnalyzer:
    def __init__(self, recordings_root="recordings", model_path="yolo11n.pt"):
        """
        初始化 YOLO 影片分析器
        
        Args:
            recordings_root: 錄製影片的根目錄
            model_path: YOLO 模型路徑 (可使用 yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        """
        self.recordings_root = recordings_root
        self.model_path = model_path
        self.output_root = "yolo_analysis"
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 建立輸出目錄
        os.makedirs(self.output_root, exist_ok=True)
        
        print(f"使用設備: {self.device}")
        
    def load_model(self):
        """載入 YOLO 模型"""
        try:
            print(f"正在載入 YOLO 模型: {self.model_path}")
            # 如果模型檔案不存在，會自動下載
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print("✓ YOLO 模型載入成功")
            return True
        except Exception as e:
            print(f"✗ 載入 YOLO 模型失敗: {e}")
            return False
    
    def get_video_files(self):
        """掃描錄製目錄，獲取所有影片檔案"""
        video_files = {}
        
        if not os.path.exists(self.recordings_root):
            print(f"錄製目錄不存在: {self.recordings_root}")
            return video_files
        
        # 支援的影片格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        # 遍歷所有子目錄（地點）
        for location_dir in os.listdir(self.recordings_root):
            location_path = os.path.join(self.recordings_root, location_dir)
            
            if os.path.isdir(location_path):
                location_videos = []
                
                # 獲取該地點的所有影片檔案
                for ext in video_extensions:
                    pattern = os.path.join(location_path, f"*{ext}")
                    location_videos.extend(glob.glob(pattern))
                
                if location_videos:
                    # 按檔案修改時間排序
                    location_videos.sort(key=lambda x: os.path.getmtime(x))
                    video_files[location_dir] = location_videos
                    print(f"找到 {location_dir}: {len(location_videos)} 個影片檔案")
        
        return video_files
    
    def analyze_single_video(self, video_path, location):
        """分析單一影片檔案"""
        print(f"開始分析影片: {video_path}")
        
        # 建立該地點的輸出目錄
        location_output_dir = os.path.join(self.output_root, location)
        os.makedirs(location_output_dir, exist_ok=True)
        
        # 建立子目錄
        frames_dir = os.path.join(location_output_dir, "detected_frames")
        videos_dir = os.path.join(location_output_dir, "annotated_videos")
        data_dir = os.path.join(location_output_dir, "detection_data")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 獲取影片檔名（不含路徑和副檔名）
        video_filename = Path(video_path).stem
        
        try:
            # 開啟影片
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"無法開啟影片: {video_path}")
                return
            
            # 獲取影片屬性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"影片資訊 - 解析度: {width}x{height}, FPS: {fps}, 總幀數: {total_frames}")
            
            # 建立輸出影片寫入器
            output_video_path = os.path.join(videos_dir, f"{video_filename}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 儲存偵測結果的列表
            detections_data = []
            frame_count = 0
            detection_count = 0
            
            print(f"開始處理影片幀...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 使用 YOLO 進行偵測
                results = self.model(frame, verbose=False)
                
                # 處理偵測結果
                frame_detections = []
                annotated_frame = frame.copy()
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 獲取邊界框座標
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[class_id]
                            
                            # 只保存信心度大於 0.5 的偵測結果
                            if confidence > 0.5:
                                detection_count += 1
                                
                                # 記錄偵測資料
                                detection_info = {
                                    'frame_number': frame_count,
                                    'timestamp': frame_count / fps,  # 秒數
                                    'class_name': class_name,
                                    'confidence': float(confidence),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                }
                                frame_detections.append(detection_info)
                                
                                # 在影片幀上繪製邊界框和標籤
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                label = f"{class_name}: {confidence:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # 如果有偵測到物件，儲存該幀
                if frame_detections:
                    detections_data.extend(frame_detections)
                    
                    # 每10幀或有重要偵測時儲存關鍵幀
                    if frame_count % 30 == 0 or len(frame_detections) > 3:  # 每秒或多個物件時儲存
                        frame_filename = f"{video_filename}_frame_{frame_count:06d}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, annotated_frame)
                
                # 寫入標註後的影片
                out.write(annotated_frame)
                
                # 顯示處理進度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"{location} - 處理進度: {progress:.1f}% ({frame_count}/{total_frames}), 偵測數: {detection_count}")
            
            # 清理資源
            cap.release()
            out.release()
            
            # 儲存偵測資料為 JSON 和 CSV
            if detections_data:
                # JSON 格式
                json_path = os.path.join(data_dir, f"{video_filename}_detections.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(detections_data, f, indent=2, ensure_ascii=False)
                
                # CSV 格式
                csv_path = os.path.join(data_dir, f"{video_filename}_detections.csv")
                df = pd.DataFrame(detections_data)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                
                # 建立偵測摘要
                summary = self.create_detection_summary(detections_data, video_filename, total_frames, fps)
                summary_path = os.path.join(data_dir, f"{video_filename}_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                print(f"✓ 完成分析: {video_path}")
                print(f"  - 總偵測數: {detection_count}")
                print(f"  - 輸出影片: {output_video_path}")
                print(f"  - 偵測資料: {json_path}")
                print(f"  - 摘要資料: {summary_path}")
            else:
                print(f"✓ 完成分析: {video_path} (未偵測到物件)")
        
        except Exception as e:
            print(f"分析影片時發生錯誤 ({video_path}): {e}")
    
    def create_detection_summary(self, detections_data, video_filename, total_frames, fps):
        """建立偵測結果摘要"""
        if not detections_data:
            return {}
        
        df = pd.DataFrame(detections_data)
        
        # 基本統計
        class_counts = df['class_name'].value_counts().to_dict()
        avg_confidence = df['confidence'].mean()
        max_confidence = df['confidence'].max()
        min_confidence = df['confidence'].min()
        
        # 時間統計
        detection_duration = df['timestamp'].max() - df['timestamp'].min()
        detection_frames = len(df['frame_number'].unique())
        
        summary = {
            'video_filename': video_filename,
            'analysis_time': datetime.now().isoformat(),
            'video_stats': {
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': total_frames / fps
            },
            'detection_stats': {
                'total_detections': len(detections_data),
                'unique_frames_with_detection': detection_frames,
                'detection_coverage_percentage': (detection_frames / total_frames) * 100,
                'detection_duration_seconds': float(detection_duration),
                'class_counts': class_counts,
                'confidence_stats': {
                    'average': float(avg_confidence),
                    'maximum': float(max_confidence),
                    'minimum': float(min_confidence)
                }
            }
        }
        
        return summary
    
    def analyze_location_videos(self, location, video_files):
        """分析某個地點的所有影片"""
        print(f"\n開始分析地點: {location} ({len(video_files)} 個影片)")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] 處理 {location} 的影片...")
            self.analyze_single_video(video_path, location)
    
    def start_analysis(self, parallel=False, max_workers=None):
        """開始分析所有影片"""
        if not self.load_model():
            return
        
        # 獲取所有影片檔案
        video_files = self.get_video_files()
        
        if not video_files:
            print("沒有找到影片檔案")
            return
        
        print(f"\n找到 {len(video_files)} 個地點的影片")
        
        if parallel and len(video_files) > 1:
            # 平行處理多個地點
            if max_workers is None:
                max_workers = min(len(video_files), multiprocessing.cpu_count())
            
            print(f"使用平行處理，工作進程數: {max_workers}")
            
            with multiprocessing.Pool(max_workers) as pool:
                tasks = [(location, videos) for location, videos in video_files.items()]
                pool.starmap(self.analyze_location_videos, tasks)
        else:
            # 序列處理
            for location, videos in video_files.items():
                self.analyze_location_videos(location, videos)
        
        print("\n🎉 所有影片分析完成!")
        self.generate_overall_report(video_files)
    
    def generate_overall_report(self, video_files):
        """產生整體分析報告"""
        print("\n正在產生整體報告...")
        
        overall_stats = {
            'analysis_time': datetime.now().isoformat(),
            'total_locations': len(video_files),
            'total_videos': sum(len(videos) for videos in video_files.values()),
            'locations': {}
        }
        
        for location in video_files.keys():
            location_dir = os.path.join(self.output_root, location, "detection_data")
            if os.path.exists(location_dir):
                summary_files = glob.glob(os.path.join(location_dir, "*_summary.json"))
                
                location_stats = {
                    'video_count': len(summary_files),
                    'total_detections': 0,
                    'class_distribution': {}
                }
                
                for summary_file in summary_files:
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                            
                        detection_stats = summary.get('detection_stats', {})
                        location_stats['total_detections'] += detection_stats.get('total_detections', 0)
                        
                        class_counts = detection_stats.get('class_counts', {})
                        for class_name, count in class_counts.items():
                            location_stats['class_distribution'][class_name] = \
                                location_stats['class_distribution'].get(class_name, 0) + count
                    
                    except Exception as e:
                        print(f"讀取摘要檔案時發生錯誤 ({summary_file}): {e}")
                
                overall_stats['locations'][location] = location_stats
        
        # 儲存整體報告
        report_path = os.path.join(self.output_root, "overall_analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 整體報告已儲存: {report_path}")

def main():
    """主程式"""
    print("=== YOLOv11 影片分析系統 ===")
    
    # 設定參數
    recordings_dir = "recordings"  # 錄製影片的目錄
    model_name = "yolo12n.pt"     # 可選: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    
    # 檢查錄製目錄是否存在
    if not os.path.exists(recordings_dir):
        print(f"錄製目錄不存在: {recordings_dir}")
        print("請確保已經執行過影片爬取程式，並且有錄製的影片檔案。")
        return
    
    # 建立分析器
    analyzer = YOLOVideoAnalyzer(recordings_dir, model_name)
    
    # 詢問使用者是否要平行處理
    use_parallel = input("是否使用平行處理？(y/n, 預設: n): ").lower().strip() == 'y'
    
    if use_parallel:
        max_workers = input(f"輸入工作進程數 (1-{multiprocessing.cpu_count()}, 預設: 自動): ").strip()
        max_workers = int(max_workers) if max_workers.isdigit() else None
    else:
        max_workers = None
    
    print(f"\n使用模型: {model_name}")
    print(f"處理模式: {'平行處理' if use_parallel else '序列處理'}")
    print(f"輸入目錄: {recordings_dir}")
    print(f"輸出目錄: yolo_analysis")
    
    input("\n按 Enter 開始分析...")
    
    # 開始分析
    start_time = time.time()
    analyzer.start_analysis(parallel=use_parallel, max_workers=max_workers)
    end_time = time.time()
    
    print(f"\n總處理時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # Windows 相容性設定
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
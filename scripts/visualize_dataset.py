import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 설정 및 src 모듈 경로 추가
file_path = Path(__file__).resolve()
project_root = file_path.parents[1]
sys.path.append(str(project_root / "src"))

from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset

class DatasetViewer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print(f"총 {len(dataset)}개의 샘플이 로드되었습니다.")
        print("조작법: [n/Space] 다음, [p] 이전, [q/ESC] 종료")
        
        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        data = self.dataset[self.idx]
        
        # Map
        map_tensor = data['map']
        map_img = map_tensor.squeeze().numpy()
        
        # Matplotlib imshow: (0,0) is top-left by default.
        map_img[1:-1, 1:-1] = 1
        
        self.ax.imshow(map_img, cmap='gray', vmin=0, vmax=1, origin='upper')
        
        # Coordinates
        W, H = 224, 224
        start = data['start'].numpy() * W
        goal = data['goal'].numpy() * W
        gt = data['gt'].numpy() * W
        
        # Scatter
        # self.ax.scatter(start[0], start[1], c='red', s=100, label='Start', edgecolors='white', zorder=5)
        # self.ax.scatter(goal[0], goal[1], c='blue', s=100, label='Goal', edgecolors='white', zorder=5)
        self.ax.scatter(gt[:, 0], gt[:, 1], c='green', s=15, label='GT', alpha=0.6, zorder=3)
        
        self.ax.set_title(f"ID: {self.idx}/{len(self.dataset)}")
        self.ax.legend(loc='upper right')
        
        # Remove axis ticks for cleaner view
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ['n', ' ', 'right']:
            self.idx = (self.idx + 1) % len(self.dataset)
            self.update_plot()
        elif event.key in ['p', 'left']:
            self.idx = (self.idx - 1) % len(self.dataset)
            self.update_plot()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

def main():
    # 데이터셋 로드 (train split)
    try:
        dataset = RLNFDataset(dataset_root_path="data", split="train")
    except Exception as e:
        print(f"데이터셋을 로드하는 중 오류가 발생했습니다: {e}")
        return

    DatasetViewer(dataset)

if __name__ == "__main__":
    main()
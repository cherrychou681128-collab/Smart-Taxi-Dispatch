import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. 路徑與基礎設定 (維持原樣)
# ==========================================
BASE_DIR = r"C:\Users\a0939\Downloads\jessnew\jessnew\data"
SHP_PATH = os.path.join(BASE_DIR, "taxi_zones.shp")
DATA_PATH = "plot_data.npz"

LAT_MIN, LAT_MAX = 40.496, 40.915
LON_MIN, LON_MAX = -74.255, -73.700

try:
    data = np.load(DATA_PATH)
    y_true, y_pred = data["y_true"], data["y_pred"]
    nyc_map = gpd.read_file(SHP_PATH)
    if nyc_map.crs is None: nyc_map.set_crs(epsg=2263, inplace=True)
    nyc_map = nyc_map.to_crs(epsg=4326)
    
    # 預先計算中心點
    nyc_map['centroid'] = nyc_map.geometry.centroid
    def get_grid_pos(point):
        j = int((point.x - LON_MIN) / (LON_MAX - LON_MIN) * 32)
        i = int((LAT_MAX - point.y) / (LAT_MAX - LAT_MIN) * 32)
        return i, j
    nyc_map['grid_pos'] = nyc_map['centroid'].apply(get_grid_pos)
except Exception as e:
    print(f"❌ 啟動失敗: {e}"); exit()

class NYCMapApp:
    def __init__(self, y_true, y_pred, nyc_map):
        self.y_true, self.y_pred, self.nyc_map = y_true, y_pred, nyc_map
        self.index = 0
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # --- 關鍵修正：預先建立固定大小的 Colorbar 區域 ---
        self.cbar_axes = []
        for ax in self.axes:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1) # 在右邊挖一個固定寬度的洞
            self.cbar_axes.append(cax)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()
        plt.show()

    def update_plot(self):
        true_matrix = self.y_true[self.index, 0]
        pred_matrix = self.y_pred[self.index, 0]
        
        # 映射數值
        def map_val(pos, mat):
            i, j = pos
            return mat[i, j] if 0 <= i < 32 and 0 <= j < 32 else 0
        
        self.nyc_map['val_true'] = self.nyc_map['grid_pos'].apply(lambda p: map_val(p, true_matrix))
        self.nyc_map['val_pred'] = self.nyc_map['grid_pos'].apply(lambda p: map_val(p, pred_matrix))

        vmax = np.percentile(true_matrix, 98) or 1
        rmse = np.sqrt(mean_squared_error(true_matrix, pred_matrix))

        configs = [
            {'col': 'val_true', 'cmap': 'YlGnBu', 'title': 'Actual Demand'},
            {'col': 'val_pred', 'cmap': 'OrRd', 'title': f'Prediction (RMSE:{rmse:.4f})'}
        ]

        for i in range(2):
            ax = self.axes[i]
            cax = self.cbar_axes[i]
            conf = configs[i]
            
            ax.clear()
            cax.clear() # 只清除內容，不刪除 Axes

            # 畫地圖
            self.nyc_map.plot(ax=ax, color='#222222', edgecolor='#444444', linewidth=0.5)
            self.nyc_map.plot(column=conf['col'], ax=ax, cmap=conf['cmap'], 
                              vmin=0, vmax=vmax, edgecolor='black', linewidth=0.2)
            
            # 更新顏色條 (畫在預留的 cax 上)
            sm = plt.cm.ScalarMappable(cmap=conf['cmap'], norm=plt.Normalize(vmin=0, vmax=vmax))
            self.fig.colorbar(sm, cax=cax) # 指定 cax，不會擠壓地圖空間
            
            ax.set_title(f"{conf['title']}\nIndex: {self.index}", color='white', fontsize=14)
            ax.set_axis_off()
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right': self.index = (self.index + 1) % len(self.y_true)
        elif event.key == 'left': self.index = (self.index - 1) % len(self.y_true)
        self.update_plot()

if __name__ == "__main__":
    NYCMapApp(y_true, y_pred, nyc_map)
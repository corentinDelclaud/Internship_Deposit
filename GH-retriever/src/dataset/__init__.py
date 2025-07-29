from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.scene_graphs import SceneGraphsDataset
from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset
from src.dataset.webqsp import WebQSPDataset
from src.dataset.webqsp_baseline import WebQSPBaselineDataset
from src.dataset.expla_graphs_khop import ExplaGraphsKhopDataset
from src.dataset.expla_graphs_khop_baseline import ExplaGraphsKhopBaselineDataset
from src.dataset.kg_graphs_v3 import KGGraphsDataset


load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'expla_graphs_khop': ExplaGraphsKhopDataset,
    'expla_graphs_khop_baseline': ExplaGraphsKhopBaselineDataset,
    'scene_graphs': SceneGraphsDataset,
    'scene_graphs_baseline': SceneGraphsBaselineDataset,
    'webqsp': WebQSPDataset,
    'webqsp_baseline': WebQSPBaselineDataset,
    'kg': KGGraphsDataset,
}

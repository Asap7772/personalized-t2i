import datasets
ds = datasets.load_dataset('yuvalkirstain/pickapic_v2')
ds.save_to_disk('downloaded_pickapic_v2')
import numpy as np
from PIL import Image, ImageEnhance
import os
# %%



azure_color_path = r'C:\Users\atlas\Desktop\azurecolor-1648243916085.jpg'
azure_depth_path = r'C:\Users\atlas\Desktop\azuredepth-1648243916086.png'
azure_ir_path= r'C:\Users\atlas\Desktop\azureir-1648243916086.png'

pg_path = r'C:\Users\atlas\Desktop\pgraw-19129366_1648244149278_0.jpg'

thermal_path = r'C:\Users\atlas\Desktop\thermal-1648243915320.png'
# thermal_path = r'C:\Users\atlas\Desktop\thermal-1648243915467.png'


color = Image.open(azure_color_path)
color_img = np.array(color)


depth = Image.open(azure_depth_path)
depth_img = np.array(depth).astype(np.uint8)


ir = Image.open(azure_ir_path)
ir_img = np.array(ir).astype(np.uint8)


pg = Image.open(pg_path)
pg_img = np.array(pg)


thermal = Image.open(thermal_path)
print(thermal.mode)
# thermal = thermal.convert('RGB')
thermal.show()

thermal_img = np.asarray(thermal).astype(np.uint8)
# thermal_img = thermal_img / 255


# ir = ir.convert('RGB')
# ir.save(r'C:\Users\atlas\Desktop\ir_test.png')
# ir.show()


# thermal = np.array(thermal)
# thermal = thermal / 255.0
thermal = Image.fromarray(thermal_img)
print(thermal.mode)
thermal.show()
# thermal.save(r'C:\Users\atlas\Desktop\thermal_test.png')

# %%
root = r'E:\calibration\PC005__169_254_47_202'

















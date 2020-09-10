import os
import shutil

BASE_PATH = 'data/zoomed/h7-29/'

channel_dirs = next(os.walk(BASE_PATH))[1]

perids = set()
for channel_dir in channel_dirs:
    (_, _, filenames) = next(os.walk(os.path.join(BASE_PATH, channel_dir)))

    for i, filename in enumerate(filenames):
        zoom_idx = filename.find('Zoom')

        if zoom_idx != -1:
            filename = filename[:zoom_idx - 1] + \
                filename[zoom_idx + len('ZoomX'):]
        cell_num_idx = filename.find('.')
        perid = filename[:cell_num_idx + 4]
        perids.add(perid)

os.mkdir(os.path.join(BASE_PATH, 'perids'))
for perid in perids:
    perid_path = os.path.join(BASE_PATH, 'perids', perid)
    os.mkdir(perid_path)
    for channel_dir in channel_dirs:
        (_, _, filenames) = next(os.walk(os.path.join(BASE_PATH, channel_dir)))
        perid_channel_path = os.path.join(perid_path, channel_dir)
        os.mkdir(perid_channel_path)
        for filename in filenames:
            perid_main, cell_num = perid.split('.')
            if perid_main in filename and cell_num in filename:
                shutil.copy2(os.path.join(BASE_PATH, channel_dir, filename),
                             perid_channel_path)

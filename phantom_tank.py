from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np
import numba
import io
import time
import os
import base64
import uuid
from collections import deque

DOWNLOAD_CACHE = {}
CACHE_EXPIRE = 60 * 5  # 5分钟缓存
CACHE_QUEUE = deque(maxlen=100)  # 最多缓存100个

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

class TurboTankGenerator:
    def __init__(self, img1, img2):
        img1_pil = self._auto_rotate(img1)
        img2_pil = self._auto_rotate(img2)
        
        # 预处理为灰度图并限制尺寸
        img1_np = self._preprocess(img1_pil)
        img2_np = self._preprocess(img2_pil)
        
        # 统一尺寸
        target_h, target_w = self._get_target_size(img1_np, img2_np)
        self.img_a = self._resize_with_padding(img1_np, target_w, target_h)
        self.img_b = self._resize_with_padding(img2_np, target_w, target_h)

    def _auto_rotate(self, img):
        try:
            exif = img._getexif()
            if exif:
                orientation = exif.get(0x0112)
                method = {
                    2: Image.FLIP_LEFT_RIGHT,
                    3: Image.ROTATE_180,
                    4: Image.FLIP_TOP_BOTTOM,
                    5: Image.TRANSPOSE,
                    6: Image.ROTATE_270,
                    7: Image.TRANSVERSE,
                    8: Image.ROTATE_90
                }.get(orientation)
                if method:
                    return img.transpose(method)
        except:
            pass
        return img
    
    def _preprocess(self, img):
        """转为灰度图,限制3000"""
        img = img.convert('L')
        if max(img.size) > 3000:
            ratio = 3000 / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32)
    
    def _get_target_size(self, a, b):
        return (
            max(a.shape[0], b.shape[0]),
            max(a.shape[1], b.shape[1])
        )
    
    def _resize_with_padding(self, img_np, target_w, target_h):
        """黑色填充"""
        img = Image.fromarray(img_np.astype(np.uint8))
        ratio = min(target_w / img.width, target_h / img.height)
        new_w, new_h = int(img.width * ratio), int(img.height * ratio)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new('L', (target_w, target_h), 0)
        new_img.paste(resized, ((target_w - new_w)//2, (target_h - new_h)//2))
        return np.array(new_img, dtype=np.float32)
    
    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def _core_algorithm(a, b):
        h, w = a.shape
        out = np.zeros((h, w, 4), dtype=np.uint8)
        for i in numba.prange(h):
            for j in numba.prange(w):
                va = a[i, j]
                vb = b[i, j]
                if va + 20 > vb:
                    vb = min(vb + (va + 20 - vb) * 0.8, 255)
                alpha = 255 - vb + va + 1
                color = (va * 255) / alpha
                out[i, j] = (color, color, color, alpha)
        return out

    def generate(self):
        result = self._core_algorithm(self.img_a, self.img_b)
        return Image.fromarray(result, 'RGBA')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        file_a = request.files['image_a']
        file_b = request.files['image_b']
        if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
            return "仅支持JPEG/PNG图片", 400
        
        img_a = Image.open(io.BytesIO(file_a.read()))
        img_b = Image.open(io.BytesIO(file_b.read()))
        
        generator = TurboTankGenerator(img_a, img_b)
        result = generator.generate()
        
        preview = result.copy()
        preview.thumbnail((400, 400), Image.Resampling.LANCZOS)
        preview_buf = io.BytesIO()
        preview.save(preview_buf, 'PNG')
        preview_b64 = base64.b64encode(preview_buf.getvalue()).decode()
        
        # 保存
        output_buf = io.BytesIO()
        result.save(output_buf, 'PNG', optimize=True, compress_level=3)
        output_buf.seek(0)
        
        token = str(uuid.uuid4())
        DOWNLOAD_CACHE[token] = {
            'data': output_buf.getvalue(),
            'timestamp': time.time()
        }
        CACHE_QUEUE.append(token)
        
        return {'preview': preview_b64, 'token': token}, 200
    except Exception as e:
        app.logger.error(f"生成失败: {str(e)}")
        return {'error': str(e)}, 500

@app.route('/download/<token>')
def download(token):
    # 清理过期缓存
    now = time.time()
    while CACHE_QUEUE and now - DOWNLOAD_CACHE[CACHE_QUEUE[0]]['timestamp'] > CACHE_EXPIRE:
        old_token = CACHE_QUEUE.popleft()
        del DOWNLOAD_CACHE[old_token]
    
    data = DOWNLOAD_CACHE.get(token, {}).get('data')
    if not data:
        return "下载链接已过期", 404
    
    return send_file(
        io.BytesIO(data),
        mimetype='image/png',
        as_attachment=True,
        download_name='phantom_tank.png'
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
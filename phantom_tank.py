from flask import Flask, request, render_template, send_file, abort
from PIL import Image
import numpy as np
import numba
import io
import time
import os
import base64
import uuid
import logging
from collections import deque,defaultdict
from datetime import datetime

DEFAULT_CONFIG = {
    "file_size_limit": 30,
    "max_requests_per_minute": 10,
}
def load_config_from_env():
    """从环境变量加载配置，缺失时回退到默认值"""
    config = DEFAULT_CONFIG.copy()  # 初始化为默认配置
    if "FILE_SIZE_LIMIT" in os.environ:
        config["file_size_limit"] = os.environ["FILE_SIZE_LIMIT"]
    if "MAX_REQUESTS_PER_MINUTE" in os.environ:
        config["max_requests_per_minute"] = int(os.environ["MAX_REQUESTS_PER_MINUTE"])
    return config
CONFIG = load_config_from_env()

required_dirs = ['logs', 'images'] 
for dir_name in required_dirs:
    os.makedirs(dir_name, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
REQUEST_COUNTER = defaultdict(list)

MAX_REQUESTS_PER_MINUTE = CONFIG['max_requests_per_minute']
FILE_SIZE_LIMIT = CONFIG['file_size_limit']
DOWNLOAD_CACHE = {}
CACHE_EXPIRE = 60 * 5 
CACHE_QUEUE = deque(maxlen=100) 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = FILE_SIZE_LIMIT * 1024 * 1024

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
    
@app.errorhandler(429)
def handle_rate_limit(error):
    return f"ERROR:{error.code}:{error.description}", error.code
@app.before_request
def check_rate_limit():
    if request.path == '/generate':
        client_ip = request.remote_addr
        now = time.time()
        # 清理超过1分钟的记录
        REQUEST_COUNTER[client_ip] = [t for t in REQUEST_COUNTER[client_ip] if now - t < 60]
        # 检查是否超限
        if len(REQUEST_COUNTER[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
            app.logger.warning(f"IP {client_ip} 触发速率限制")
            abort(429, description="请求过于频繁，请稍后再试")
        # 记录本次请求时间
        REQUEST_COUNTER[client_ip].append(now)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        file_a = request.files.get('image_a')
        file_b = request.files.get('image_b')
    
        if not (file_a and file_b):
            app.logger.error(f"缺少文件上传，客户端IP: {request.remote_addr}")
            return "请选择两张图片进行生成", 400
            
        if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
            app.logger.error(f"非法文件类型，客户端IP: {request.remote_addr}")
            return "仅支持JPEG/PNG图片", 400

        token = str(uuid.uuid4())
        logging.info(f"开始处理请求，Token: {token}")

        base_dirs = {
            'sideA': os.path.join('images', token, 'sideA'),
            'sideB': os.path.join('images', token, 'sideB'),
            'output': os.path.join('images', token, 'output')
        }
        for d in base_dirs.values():
            os.makedirs(d, exist_ok=True)

        def save_upload(file_obj, side):
            ext = file_obj.filename.rsplit('.', 1)[1].lower()
            save_path = os.path.join(base_dirs[side], f'original.{ext}')
            file_data = file_obj.read()
            with open(save_path, 'wb') as f:
                f.write(file_data)
            return file_data

        file_a_data = save_upload(file_a, 'sideA')
        file_b_data = save_upload(file_b, 'sideB')

        # 处理图像
        img_a = Image.open(io.BytesIO(file_a_data))
        img_b = Image.open(io.BytesIO(file_b_data))
        
        generator = TurboTankGenerator(img_a, img_b)
        result = generator.generate()

        output_path = os.path.join(base_dirs['output'], 'result.png')
        result.save(output_path, 'PNG', optimize=True, compress_level=3)

        preview_buf = io.BytesIO()
        result.thumbnail((400, 400), Image.Resampling.LANCZOS)
        result.save(preview_buf, 'PNG')
        preview_b64 = base64.b64encode(preview_buf.getvalue()).decode()

        # 缓存下载文件
        output_buf = io.BytesIO()
        result.save(output_buf, 'PNG')
        DOWNLOAD_CACHE[token] = {
            'data': output_buf.getvalue(),
            'timestamp': time.time()
        }
        CACHE_QUEUE.append(token)

        app.logger.info(f"处理成功，Token: {token}")
        return {'preview': preview_b64, 'token': token}, 200

    except Exception as e:
        app.logger.error(f"处理失败: {str(e)}")
        return "处理失败: "+str(e), 500

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
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"phantom_tank_{timestamp}.png"
    return send_file(
        io.BytesIO(data),
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
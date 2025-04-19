# # phantom_tank 幻影坦克

在线生成

## 🔧 部署说明

### 使用python启动，所需依赖在requirements.txt文件内

文件结构

```
vnstat-assist ->总目录
  -phantom_tank.py -> python服务，后端处理和前端挂载
  -requirements.txt -> 依赖文件
  -Dockerfile -> 打包配置
  -docker-compose.yml -> docker构建配置
  -templates
    -index.html -> 前端页面
  -conf
    -supervisord.conf -> supervisord配置
```

### docker-compose配置

```
version: '3'
services:
  phantom_tank:
    image: phantom_tank:latest
    container_name: phantom_tank
    restart: unless-stopped
    ports:
      - "5000:5000"
```

## 🧩 界面截图

![1](screenshots/1.png)

![2](screenshots/2.png)

![2](screenshots/3.png)

![2](screenshots/4.png)

### ⚠️说明：示例用图来自[FKEY](https://weibo.com/u/2182437790)老师，仅演示不做其他用途
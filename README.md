人体姿态识别项目说明
山东大学（威海）
18 数据科学 徐潇涵

本项目开发了一款基于微信小程序的动作识别系统，旨在实时识别并计数用户手持手机完成的特定健身动作。

系统聚焦于徒手侧平举、前后交叉小跳、开合跳及半蹲四种常见动作。核心技术在于：当测试者使用数据采集小程序时，小程序实时采集设备内置传感器（加速度计与陀螺仪）提供的六轴运动数据。随后，系统融合随机森林机器学习模型进行动作分类，并结合实时波峰检测算法精准捕捉动作周期，将模型部署至云服务器并接入小程序端，最终实现对用户动作的实时识别与自动计数。

b站视频链接：[https://www.bilibili.com/video/BV1oT4y1L7a8](https://www.bilibili.com/video/BV1Rp4y1i7kT/?buvid=Y2456623E2EA83D946FF9C2E05534577F987&from_spmid=main.space.0.0&is_story_h5=false&mid=b589vN1b4YB5FLOGeFcmNA%3D%3D&p=1&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=6E9FAD03-5C24-48B7-B5D0-2B9D34DEBD53&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749210707&unique_k=zTulsrk&up_id=398313570)

data/ 包含了本项目所收集的全部数据，已经将数据中安卓数据与苹果数据提前分开

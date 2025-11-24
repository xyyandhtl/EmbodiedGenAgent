给我根据图片场景，定义 yolov8-world 的开放语义英文 object list。
每个语义一行，并加对齐#号的中文注释在每行后面, 我后面要放进 txt 文件中，要求不要用连接符，复合词可以用 PascalCase 的写法。
不要包含表示地面相关的语义，比如 floor, ground, street 等等，但像人行道这种附着在地面的类似语义是需要的。
总类别不要太多，20类左右即可。

```txt
Human                     # 行人
Car                       # 小汽车
Truck                     # 卡车
StreetLamp                # 路灯
Bench                     # 长椅
TrashBin                  # 垃圾桶
TrafficCone               # 交通锥
Building                  # 建筑
Window                    # 窗户
Door                      # 门
Awning                    # 遮阳棚
Sidewalk                  # 人行道
Fence                     # 栅栏
FireHydrant               # 消防栓
SignBoard                 # 标志牌
Graffiti                  # 涂鸦
Mailbox                   # 邮箱
Shop                      # 店铺遮篷
Restaurant                # 餐馆
ConstructionBarrier       # 施工围栏
TrashBag                  # 垃圾袋
Mailbox                   # 信箱
UtilityPole               # 电线杆
```

在列出总类别以后，再在这些语义中分类为高移动性和低移动性。其中高移动性仅包含人、动物、车、垃圾桶等这些极容易移动的 object，不确定的都归入低移动性。
参考以下格式：
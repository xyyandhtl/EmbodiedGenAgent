给我根据图片场景，定义 yolov8-world 的开放语义英文 object list，每个语义一行，我后面要放进 txt 文件中，要求不要用连接符，复合词可以用 PascalCase 的写法。
然后再在这些语义中分类为高移动性和低移动性。其中高移动性仅包含人、动物、车、垃圾桶等这些极容易移动的 object，不确定的都归入低移动性。
不要包含表示地面相关的语义，比如 floor, ground, street 等等，但像人行道这种附着在地面的类似语义是需要的。
总类别不要太多，20类左右即可。
参考以下格式：

```txt
Building
Apartment
Window
Balcony
Door
Awning
AirConditioner
Signboard
Billboard
Shop
Restaurant
Cafe
Stairs
Gate
Fence
Sidewalk
Crosswalk
Streetlight
ElectricPole
FireHydrant
TrafficLight
TrafficSign
BusStop
TrafficCone
TrafficBarrier
ConstructionSign
Tree
Bush
Bench
Flowerpot
TrashBin
Mailbox
Car
Truck
Bicycle
Motorcycle
Person
```

```txt
# low mobility examples
lm_examples:
  - "Building"
  - "Apartment"
  - "Window"
  - "Balcony"
  - "Door"
  - "Awning"
  - "AirConditioner"
  - "Signboard"
  - "Billboard"
  - "Shop"
  - "Restaurant"
  - "Cafe"
  - "Stairs"
  - "Gate"
  - "Fence"
  - "Sidewalk"
  - "Crosswalk"
  - "Streetlight"
  - "ElectricPole"
  - "FireHydrant"
  - "TrafficLight"
  - "TrafficSign"
  - "BusStop"
  - "TrafficCone"
  - "TrafficBarrier"
  - "ConstructionSign"
  - "Tree"
  - "Bush"
  - "Bench"
  - "Flowerpot"
  - "TrashBin"
  - "Mailbox"

# high mobility examples
hm_examples:
  - "Car"
  - "Truck"
  - "Bicycle"
  - "Motorcycle"
  - "Person"

# Extra descriptions for low mobility examples
lm_descriptions:
  - "urban structures, infrastructures, and objects fixed to the ground"
  - "architectural and environmental components that rarely move"
  - "elements forming the static and stable part of the city scene"
  - "man-made or natural objects that remain stationary for long periods"
```
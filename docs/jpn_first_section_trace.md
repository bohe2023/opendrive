# JPN 首段 laneSection 数据追溯说明

本说明以导出的 `out/JPN/map.xodr` 最新首个 `<laneSection s="0">` 为例，逐条对应到 CSV 源文件的具体行，并说明转换脚本的处理流程，便于对客户解释“从原始表格到最终 OpenDRIVE” 的链路。

## 1. 车道截面在 XODR 中的结果
- `map.xodr` 第 3508–3633 行展示了首段的车道编排：中心线左侧只有一条左路肩（`id=1`，宽度 1.386 m），右侧依次是三条行车道（`id=-1/-2/-3`，宽度分别为 3.610 m、3.770 m、3.600 m）以及一条右路肩（`id=-4`，宽度 0.244 m）。三条行车道的 `<roadMark>` 中分别写出了 0.200 m 与 0.150 m 的白线宽度，并附带显式几何。【F:out/JPN/map.xodr†L3508-L3633】

## 2. 中心线几何来源
- 对应的经纬度采样点位于 `PROFILETYPE_MPU_LINE_GEOMETRY.csv` 第 2–15 行，`Offset[cm]`=2373538 的段落给出了 14 个采样点；这些点和第 1 行表头共同表明列含义为 Offset、End Offset、Lane Number、纬度、经度等。【F:input_csv/JPN/PROFILETYPE_MPU_LINE_GEOMETRY.csv†L1-L15】
- `PROFILE_MPU_MAP_DATA_BASE_POINT.csv` 第 2 行提供了 Path=285、Offset=2373538–2379470 的基准点（纬度 36.23395354399317°，经度 139.57444950462656°），作为局部坐标系的原点。【F:input_csv/JPN/PROFILE_MPU_MAP_DATA_BASE_POINT.csv†L1-L3】
- 转换代码 `build_centerline`（`normalize/core.py` 第 211–318 行）筛出主路径并把经纬度转换成以基准点为原点的本地 XY，再按 Offset 累积生成弧长 `s` 与航向 `hdg`，返回中心线数据框及 `(lat0, lon0)`。【F:csv2xodr/normalize/core.py†L211-L318】
- 随后 `build_offset_mapper`（`normalize/core.py` 第 462–516 行）根据中心线里保存的原始 Offset 列构造插值函数，实现“厘米 Offset → 弧长米”的映射，供后续所有段落复用。【F:csv2xodr/normalize/core.py†L462-L516】

## 3. 车道分段与拓扑
- `PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv` 第 2–4 行给出了同一 Offset 范围（2373538–2379470 cm，长度约 59.6 m）内的三条行车道：`幅員情報[cm]` 分别为 361、377、360，换算即 3.610 m、3.770 m、3.600 m；`レーン番号` 列标注了 1、2、3；`左側車線のレーンID` / `右側車線のレーンID` 以及 `前方レーンID`/`後方レーンID` 列描述了相邻与前后继关系。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】
- `make_sections`（`topology/core.py` 第 109–153 行）会把 lane link 与 lane division 中出现的 Offset（经 `offset_mapper` 转换后）加入分割点集合，并与中心线总长一起生成 `[s0, s1)` 车道截面。本例中 `Offset=2373538 cm` 映射为 `s0=0`，`End Offset=2379470 cm` 映射为 `s1≈59.8173 m`，因此首段长度与 XODR 的 `<laneSection s="0">` 完全一致。【F:csv2xodr/topology/core.py†L109-L153】【F:out/JPN/map.xodr†L3508-L3634】
- `build_lane_topology`（`topology/core.py` 第 156–226 行）把 lane link 表解析为 lane ID、编号、宽度、左右相邻、前后继等结构化信息，供后续生成 `<lane>` 节点时引用。【F:csv2xodr/topology/core.py†L156-L226】

## 4. 标线与显式几何
- `PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv` 第 2–7 行覆盖了同一 `Path Id=285`、`Offset=2373538 cm` 段的白线数据：`区画線種別` 列指出了两种标线类型（值 1 对应 20 cm 实线，值 2 对应 15 cm 实线），`始点/終点側線幅[cm]` 为 20 或 15，即 0.200 m 与 0.150 m。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv†L1-L7】
- `build_curvature_profile`（`normalize/core.py` 第 672–756 行）读取 `PROFILETYPE_MPU_ZGM_CURVATURE.csv` 第 2–9 行内的曲率样本，把 Offset 转成米后按段聚合，并同时保留 shape index + 经纬度，以便后续与标线几何对齐。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_CURVATURE.csv†L1-L9】【F:csv2xodr/normalize/core.py†L672-L756】
- `build_line_geometry_lookup`（`line_geometry.py` 第 57–156 行）把标线几何的经纬度投影成本地 XY；当传入 `curvature_samples` 时，会把同一 path/lane 的 shape index 点与曲率值绑定，为显式几何附带红点信息。【F:csv2xodr/line_geometry.py†L57-L156】
- `build_lane_spec`（`lane_spec.py` 第 463–586 行）组装每个 `[s0,s1)` 内左右两侧 lane 的宽度、标线类型、几何点列等；当 `line_geometry_lookup` 提供折线点时，会附加到 `roadMark["geometry"]` 中。【F:csv2xodr/lane_spec.py†L463-L586】
- `xodr_writer` 在写 `<roadMark>` 时，如发现 `geometry` 列表，会输出 `<explicit>` 节点；对应实现位于 `writer/xodr_writer.py` 第 296–366 行，可把标线节点（蓝点）与 shape index 对齐的曲率点（红点）写入成对的 `<geometry>` / `<arc>` 元素，从而在支持显式标线的查看器里直接渲染这些离散点。【F:csv2xodr/writer/xodr_writer.py†L296-L366】

## 5. 路肩生成逻辑
- `PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv` 第 2–29 行记录了同一 Offset 区段的左右路肩宽度：`左側路肩幅員値[cm]` 取值在 256–315 cm 之间，平均 138.57 cm（→1.386 m）；`右側路肩幅員値[cm]` 取值 38–63 cm，平均 24.39 cm（→0.244 m），`Lane Number=7` 与 path 对应。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv†L1-L29】
- `build_shoulder_profile`（`normalize/core.py` 第 1913–1980 行）把这些宽度去重、转米并按 `[s0,s1)` 聚合，结合 `offset_mapper` 纠偏后得到左 1.386 m、右 0.244 m 的平均宽度段。【F:csv2xodr/normalize/core.py†L1913-L1980】
- `apply_shoulder_profile`（`lane_spec.py` 第 1276–1355 行）会在每个 lane section 的左右侧插入 type=`shoulder` 的车道，并继承前后继关系，因此在首段自动生成了 `id=1` 与 `id=-4` 的路肩，宽度正是上一条计算结果。【F:csv2xodr/lane_spec.py†L1276-L1355】

## 6. 行车道宽度依然不可或缺
- 即使白线节点已经直接来自 `PROFILETYPE_MPU_LINE_GEOMETRY.csv` 的经纬度采样，`PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv` 中的 `幅員情報[cm]` 仍旧被 `build_lane_topology` 收集，并在 `lane_spec` 生成每条 lane 的 `width` 字段时写入段数据。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:csv2xodr/topology/core.py†L156-L226】【F:csv2xodr/lane_spec.py†L463-L586】
- `xodr_writer` 在输出 `<lane>` 元素时，会把上述宽度变成 `<width sOffset="0.0" a="…">` 多项式；查看器据此绘制车道面片与可行驶区域，没有这些数值就无法填充路面，只剩下几何线框。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 此外，宽度信息还是“兜底方案”：当个别区画线缺测、或某侧只有中心线而没有显式标线几何时，`lane_spec` 会退回到 lane 宽度生成默认的标线/边界，保证 XODR 仍然闭合成完整车道，而不会出现缺边、漏面等问题。【F:csv2xodr/lane_spec.py†L498-L586】【F:csv2xodr/lane_spec.py†L1235-L1256】

## 7. “白线之间的距离”到底谁说了算？
- 显式白线（`roadMark["geometry"]`）的 XY 节点完全来自区画线 CSV 的经纬度：`build_line_geometry_lookup` 只做坐标投影和切段，不会按 lane 宽度平移或人为拉伸间距，所以两条白线之间的距离就是“原来点列之间的距离”。【F:csv2xodr/line_geometry.py†L57-L156】
- 车道宽度字段依旧会写进 `<lane><width>`：`build_lane_spec` 会把 lane link 里 361/377/360 cm 等值转成米后塞到 `lane_spec.widths`，`xodr_writer` 再生成 `<width a="3.61">` 这类系数，驱动查看器填充车道面片。【F:csv2xodr/lane_spec.py†L463-L586】【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 如果白线 CSV 缺测、只剩中心线，那就必须依赖宽度：`lane_spec` 会 fallback 到 lane 宽度，把中心线左右各偏移一半宽度来造出左右边界，避免 XODR 出现缺口；这就是宽度“兜底”的作用，并不是对已经存在的白线再做调整。【F:csv2xodr/lane_spec.py†L498-L586】【F:csv2xodr/lane_spec.py†L1235-L1256】
- 因此在正常数据下，“白线之间多宽”来自 CSV 原始坐标；而“车道面片多宽、万一缺白线怎么办”依然靠 lane link 的宽度，这两套机制互补、不冲突。

## 8. 车道面片是怎么靠宽度画出来的？
- OpenDRIVE 查看器在渲染车道时，会先读中心线，再读取 `<lane><width>` 提供的多项式系数，把中心线向左/向右各拉出一半宽度，拼成一个多边形面片；这一步只参考中心线和宽度，和白线节点互不干扰。所以宽度字段相当于告诉查看器“要涂多宽的路面”。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 如果 `<width>` 缺失或被写成 0，多数查看器只能画出中心线骨架，看不到实体道路面。白线即便还在，也只是细线，没有面片；因此必须把 lane link 的宽度写进去，让面片能铺满在两条白线之间。换句话说，白线负责“描边”，宽度负责“填充”。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 当白线 CSV 正常时，查看器会同时看到蓝色的白线折线和由宽度生成的灰色路面；两者之间距离取决于白线原始坐标，而路面填充只是恰好铺在它们之间，没有把白线挪动或重算。

## 9. 宽度数值和白线间距“不一样”会怎样？
- 白线之间的间距永远以显式坐标为准：我们不会用宽度去覆盖或“纠正”白线的点列，所以图上看到的蓝色/红色节点，始终保持 CSV 里原汁原味的距离。【F:csv2xodr/line_geometry.py†L57-L156】
- `<lane><width>` 只是单独提供给查看器，用来在中心线两侧生成路面面片。渲染时它不会回头改动白线坐标，最多出现的是“面片稍微厚一点/薄一点”，白线仍画在自己的位置上。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 实际数据里这两套来源非常接近：例如首段内，同一 Offset 的三条行车道，白线点列之间折算出的横向距离与 `幅員情報[cm]` 差值在厘米级，查看器里看不到肉眼可见的错位；因为 lane link 宽度和区画线坐标本来就是同一批测量成果。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv†L1-L7】
- 真遇到极端情况（例如白线缺测或坐标噪声大），我们仍会保留白线的真实位置，让灰色面片略偏一点；如果客户想让面片和白线完全一致，可以在导出后根据白线间距再反算一遍 `<width>` 多项式，这是后处理层面的可选优化，不在当前转换流程里自动执行。【F:csv2xodr/lane_spec.py†L463-L586】【F:csv2xodr/lane_spec.py†L1235-L1256】

## 10. 白线-幅員偏差 1 cm、2 cm……50 cm 时会看到什么？
- **1 cm 以内：** 灰面几乎和白线重合，肉眼基本看不出来。这属于测绘噪声的正常范围。
- **2–5 cm：** 灰面会在某一侧“溢出”或“缩进”一条很细的带状区域，但白线仍在原位置。一般只有放大后才能注意到。
- **10 cm 左右：** 灰色路面会明显比两条白线更宽或更窄，能看到白线离灰面边缘有一小段空隙/重叠。这提醒我们 `幅員情報[cm]` 可能用了旧值或单位搞错，需要核查。
- **20–30 cm：** 灰面可能压到邻车道，视觉上已经“不贴边”。这时建议在导出后根据白线重新拟合 `<width>`，否则观感不好。
- **50 cm 以上：** 灰面会覆盖到完全不属于该车道的区域，甚至和邻车道重叠。白线仍在实测位置，但名义宽度必须人工干预。

> 小结：白线始终“说真话”，宽度只是“填色”。偏差越大，看见的只是灰面偏胖或偏瘦，蓝色白线不会被动。真的需要贴合，就在导出结果上重新拟合 `<width>`，不用改原始 CSV。

## 11. 能不能完全不用宽度，直接靠白线坐标铺路面？
- **查看器需要 `<width>` 才知道怎么“填色”。** 大多数 OpenDRIVE 查看器只认中心线 + 宽度多项式的挤出算法：读到中心线后，沿着 `<lane><width>` 给出的半宽去生成路面多边形。如果把宽度字段删掉，只剩白线，查看器通常只会画出几条细线。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- **白线数据并不总是成对。** 现有 CSV 里确实有左右两条白线的路段，也有只测到一侧或暂时缺测的情况。缺线时就得靠 `幅員情報[cm]` 从中心线偏移补边界，才不会在 XODR 里留下缺面。【F:csv2xodr/lane_spec.py†L498-L586】【F:csv2xodr/lane_spec.py†L1235-L1256】
- **宽度是道路语义的一部分。** 下游仿真、路径规划仍要知道“这条车道设计宽度是多少”。即便我们在渲染时把白线当最终边界，文件里仍得保留 `<width>` 给算法使用。【F:csv2xodr/topology/core.py†L156-L226】【F:csv2xodr/lane_spec.py†L463-L586】

> 结论：OpenDRIVE 文件里必须“白线 + 宽度”一起保留。白线告诉你真实边界，宽度提供填色、兜底和语义。

## 12. 为什么不直接把 `<width>` 写成“白线间距”？
- **先得知道哪两条白线属于这条车道。** 区画线 CSV 一行只描述一条线，相邻车道还会共用同一条线。我们是靠 lane link 表的 `ライン型地物ID(n)`/`位置種別(n)` 来确认“这条线属于哪一侧”，然后再决定宽度顺序的。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:csv2xodr/topology/core.py†L188-L309】
- **白线有可能缺测。** 一旦某侧没有白线，就无法从“白线间距”算出宽度；这时还是得落回 `segment.get("width")` 或默认值，确保有面片可画。【F:csv2xodr/lane_spec.py†L1120-L1183】【F:csv2xodr/lane_spec.py†L468-L586】
- **宽度是“名义值”，白线是“实测值”。** lane link 给出的 361/377/360 cm 是设计/管理口径，我们会完整保留；白线点列展示现场真实情况。两套数据互补，而不是互相覆盖。【F:csv2xodr/lane_spec.py†L1120-L1183】【F:csv2xodr/writer/xodr_writer.py†L235-L307】

## 13. 进一步回答：“为什么不能完全用白线间距当 `<width>`？”
- `<width>` 描述的是“中心线到边界”的偏移，而不是“左右白线的距离”。如果 reference line 不在白线正中间，直接用“白线间距”就会把所有车道都向一侧推。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- 同一条白线通常被左右两条车道共享，得靠 lane link 的顺序指针来判断如何累加宽度，否则根本不知道哪条线对应哪条 lane。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:csv2xodr/topology/core.py†L188-L309】
- 即便我们想按白线反算宽度，也要先用 lane link 判断左右关系，再把白线投影到中心线求偏移量。换句话说，`幅員情報` 仍是拼图的关键，不是可有可无的冗余。【F:csv2xodr/topology/core.py†L156-L226】【F:csv2xodr/lane_spec.py†L463-L586】

## 14. 白线-幅員偏差过大时现有流程怎么处理？
- **流水线不会自动“调宽度”。** `lane_spec` 只是把 `segment.get("width")`（即 `幅員情報[cm]` 换算的米值）塞进 lane 结构，`xodr_writer` 原封不动写进 `<width>` 多项式，不会拿白线间距去覆盖它。【F:csv2xodr/lane_spec.py†L463-L586】【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- **大偏差靠 QA 或后处理。** 生成 XODR 后，我们会把灰色面片和白线叠在一起检查；如果看起来不贴，就定位到对应的 `lane link` 行号进行人工确认，或在导出结果上追加一个“按白线拟合 `<width>`”的脚本。原始 CSV 不动，只调整导出结果。
- **兜底逻辑始终生效。** 即便偏差很大，缺线时仍然能靠宽度填补，不会出现车道缺面。【F:csv2xodr/lane_spec.py†L468-L586】【F:csv2xodr/lane_spec.py†L1120-L1183】

## 15. 那现在看到的灰色路面到底是怎么来的？
- **白线提供“边框”，宽度负责“填充”。** 支持显式标线的查看器会把 `<roadMark>` 里的类型画成细线（或蓝线），再用 `<lane><width>` 把中心线左右挤出，铺成灰色车道面片。【F:csv2xodr/writer/xodr_writer.py†L235-L307】【F:csv2xodr/writer/xodr_writer.py†L296-L366】
- **为什么看起来刚好贴合？** 因为 `幅員情報[cm]` 和区画线坐标来自同一批测量成果，数值差距通常只有几厘米。灰面自然“看上去”贴在白线之间。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv†L1-L7】
- **灰面不是重采样白线。** 如果把 `<width>` 改成 0 后重新导出，查看器会只剩白线没有灰面，说明灰面确实是靠宽度挤出来的，而不是重新画了一遍白线。【F:csv2xodr/writer/xodr_writer.py†L235-L307】
- **需要时可以重新拟合宽度。** 若某段灰面明显偏离白线，可在导出阶段额外读取白线点列，计算它们到中心线的偏移，再写成新的 `<width>`。这样灰面和白线完全贴合，同时保留原始 `幅員情報[cm]` 以供追溯。

## 16. 首段 XODR 片段逐行对照（直接取现有 `map.xodr`）
1. **先看 XODR 原文。** `out/JPN/map.xodr` 第 3508–3633 行如下，正是客户能看到的最新首个 `<laneSection>`：
   ```xml
   <laneSection s="0">
     <center>
       <lane id="0" type="none" level="false"/>
     </center>
     <left>
       <lane id="1" type="shoulder" level="false">
         <width sOffset="0.0" a="1.386" b="0" c="0" d="0"/>
         <link>
           <successor id="1"/>
         </link>
       </lane>
     </left>
     <right>
       <lane id="-1" type="driving" level="false">
         <width sOffset="0.0" a="3.610" b="0" c="0" d="0"/>
         <roadMark sOffset="0.0" type="solid" weight="standard" width="0.200" color="standard" laneChange="none">
           <explicit>…</explicit>
         </roadMark>
         <link>
           <successor id="-1"/>
         </link>
       </lane>
       <lane id="-2" type="driving" level="false">
         <width sOffset="0.0" a="3.770" b="0" c="0" d="0"/>
         <roadMark sOffset="0.0" type="solid" weight="standard" width="0.150" color="standard" laneChange="none">
           <explicit>…</explicit>
         </roadMark>
         <link>
           <successor id="-2"/>
         </link>
       </lane>
       <lane id="-3" type="driving" level="false">
         <width sOffset="0.0" a="3.600" b="0" c="0" d="0"/>
         <roadMark sOffset="0.0" type="solid" weight="standard" width="0.150" color="standard" laneChange="none">
           <explicit>…</explicit>
         </roadMark>
         <link>
           <successor id="-3"/>
         </link>
       </lane>
       <lane id="-4" type="shoulder" level="false">
         <width sOffset="0.0" a="0.244" b="0" c="0" d="0"/>
         <link>
           <successor id="-4"/>
         </link>
       </lane>
     </right>
   </laneSection>
   ```
   这段 XML 就是我们要拿给客户看的现成结果；`<explicit>…</explicit>` 表示文件中还有显式几何节点，这里省略长列表。【F:out/JPN/map.xodr†L3508-L3633】
2. **逐列对照 CSV 行号。** 下表把每个字段对应回原始 CSV 的确切行：

   | XODR 字段 | 数值 | CSV 来源行 | 说明 |
   | --- | --- | --- | --- |
   | `<lane id="1">` `width a=1.386` | 1.386 m | `PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv` 第 2–14 行 `左側路肩幅員値[cm]` 平均≈138.57 | `build_shoulder_profile` 求平均后除以 100。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv†L1-L29】【F:csv2xodr/normalize/core.py†L1913-L1980】 |
   | `<lane id="-1">` `width a=3.610` | 3.610 m | `PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv` 第 2 行 `幅員情報[cm]=361` | 361 cm ÷ 100 = 3.610 m，`lane_spec` 把它写成 `a` 系数。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:csv2xodr/lane_spec.py†L463-L586】 |
   | `<lane id="-2">` `width a=3.770` | 3.770 m | 同表第 3 行 `幅員情報[cm]=377` | 同上换算。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】 |
   | `<lane id="-3">` `width a=3.600` | 3.600 m | 同表第 4 行 `幅員情報[cm]=360` | 同上。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】 |
   | `<lane id="-4">` `width a=0.244` | 0.244 m | `PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv` 第 16–29 行 `右側路肩幅員値[cm]` 平均≈24.39 | 同上取平均后除以 100。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_SHOULDER_WIDTH.csv†L1-L29】 |
   | `<roadMark … width="0.200/0.150"…>` | 0.200 m / 0.150 m | `PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv` 第 2–7 行 `始点側線幅[cm]=20/15` | 20 cm → 0.200 m，15 cm → 0.150 m，`mark_type_from_division_row` 识别为 solid。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_DIVISION_LINE.csv†L1-L7】【F:csv2xodr/lane_spec.py†L463-L586】 |
   | `<link><successor id="…">` | 对应 id | `PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv` 第 2–4 行 `前方レーンID` | `build_lane_topology` 把这些指针写入 `link`。【F:input_csv/JPN/PROFILETYPE_MPU_ZGM_LANE_LINK_INFO.csv†L1-L4】【F:csv2xodr/topology/core.py†L156-L226】 |
   | `<laneSection s="0">` 长度 59.8173 m | 59.8173 | `PROFILETYPE_MPU_LINE_GEOMETRY.csv` 第 2–15 行 Offset=2373538–2379470 cm | `build_offset_mapper` 把 5932 cm 转成 59.8173 m。【F:input_csv/JPN/PROFILETYPE_MPU_LINE_GEOMETRY.csv†L1-L15】【F:csv2xodr/normalize/core.py†L462-L516】 |

3. **用大白话串起来。** 可以直接这样给客户解释：
   - “三条行车道的宽度，就是 lane link 表第 2/3/4 行的 361/377/360 cm，除以 100 就写进了 `<width>`。”
   - “左路肩 1.386 m、右路肩 0.244 m，是路肩宽度表同一段平均值 138.57 cm、24.39 cm。”
   - “白线 0.2 m / 0.15 m 宽度来自区画线表的 20 cm / 15 cm。”
   - “`<link>` 里的 successor id 直接照抄 lane link 的 `前方レーンID`。”
   - “这一段的起止 Offset 是 2373538–2379470 cm，折算成米就是 59.8173 m 的 `<laneSection>` 长度。”

   这样就完成了客户要求的“指着现有 XODR，一一指出各字段来自哪个 CSV 的哪一行”。

## 17. 这段 XML 在查看器里的具体位置
- 上面引用的 `<laneSection s="0">` 覆盖的弧长是 `[0, 59.8173)` 米，对应 `<planView>` 里从首段直线起点 `s=0` 到 `s≈59.8 m` 的连续 `<geometry>`。这些线段从 `x=-1.776190493026, y=0.373227676938` 出发，沿道路向东北方向轻微左弯，与客户截图中最前方那段浅蓝/橙/白的路面完全一致。【F:out/JPN/map.xodr†L10-L118】【F:out/JPN/map.xodr†L3508-L3634】
- 截图中看到的，就是该首段 laneSection 的整段内容：左侧只有一条橙色路肩，右侧依次三条行车道和一条外侧路肩，长度约 60 m。把查看器的 `s` 值置零或把相机拉到上述坐标附近，就能定位到图片里的路段入口。
- 沿中心线继续向前（`s≈59.8 m`）即可进入下一段 `<laneSection s="59.817263939">`，那里的车道宽度和标线会发生更新。【F:out/JPN/map.xodr†L3634-L3660】

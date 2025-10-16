# CSV 车道生成流程说明

## 总体流程
`pythonProject/main.py` 在转换 CSV 数据集时，会依次执行以下步骤：

1. 用 `build_centerline` 从 `LineGeometry` 表恢复道路参考线，并同时确定经纬度原点。 这一结果会被后续所有 offset 映射复用。【F:csv2xodr/csv2xodr.py†L72-L92】
2. `make_sections` 根据 `Lane Link`/`Lane Division` 表中的起讫桩号（offset）切分路段，决定每个 lane section 的纵向范围。【F:csv2xodr/topology/core.py†L110-L147】
3. `build_lane_topology` 逐行解析 `PROFILETYPE_MPU_*_LANE_LINK_INFO.csv`，提取车道编号、宽度、前后继、左右相邻等信息，形成以车道为单位的拓扑描述。【F:csv2xodr/topology/core.py†L150-L414】
4. `build_lane_spec` 把 lane section、lane topology、车道线（`Lane Division`/`Line Geometry`）等信息融合在一起，生成最终写入 XODR 的 lane 规格（类型、宽度、车道线、几何等）。【F:csv2xodr/lane_spec.py†L463-L600】

最终 `write_xodr` 会按照这些规格拼装出每个 lane 的 XML 片段，并自动补齐默认路肩等设置。【F:csv2xodr/writer/xodr_writer.py†L229-L388】

## 单条车道的来源
- `Lane Link` 表中的一行就代表一条“主车道”或功能车道（加减速、匝道等）。代码通过 `Lane ID` 与 `Lane Number` 组合出唯一标识，并读取该行的起点/终点 offset、左右邻接关系、前后继目标等字段。【F:csv2xodr/topology/core.py†L192-L323】
- 如果该行被标记为 "Accelerating"、"Decelerating" 或者 `Lane Add/Remove`，转换器会把它视作加减速/增减车道，默认跳过，不纳入主线拓扑。【F:csv2xodr/topology/core.py†L266-L312】
- 每条车道在纵向上的分段，会与 `Lane Division` 中的车道线 segment 相互匹配，从而绑定对应的道路标线类型和线宽。如果 `Line Geometry` 里能找到同 ID 的折线，还会使用这些坐标来重建车道线的空间形状，而不是单纯依赖默认宽度偏移。【F:csv2xodr/lane_spec.py†L286-L360】【F:csv2xodr/lane_spec.py†L420-L440】

因此，一条车道的几何确实是靠 “参考线 + 车道线坐标 + 宽度/偏移” 拼接出来的，只是这些数据来自多个 CSV 表联合驱动，而不是单表直接给出完整折线。

## 为什么同一路段的车道数量会有出入
- `Lane Link` 表里往往会把主线、匝道、加减速车道混在一起，使用 `Lane Type`、`Accelerating/Decelerating`、`Lane Add/Remove` 等字段区别用途。转换器只会把未被标记为加减速/增减的记录当作常规“Normal Driving Lane”。【F:csv2xodr/topology/core.py†L266-L312】
- `Lane Count` 列（如果存在）只被当作一个参考值：程序会读取 `lane_count_col` 的首行，作为“预期车道数”统计，但真正参与构图的是每条满足条件的行。也就是说，即使表头写着 "Lane Count = 5"，如果后续行被判定为匝道或增减车道，输出里仍然只会保留实际被采纳的那几条。
- 反过来，少数 CSV 文件在某些桩号区间只登记了一部分车道（例如先只登记最内侧车道，随后在下游桩号才追加其余车道）。在这些区间内，转换结果会暂时只保留已出现的那几条车道，直到新的桩号段把完整车道补齐。

## 建议的处理方式
1. **确认数据标注意图**：核对原始 CSV，确认标记为 "Accelerating"/"Decelerating"/"Lane Add/Remove" 的行是否应该视作主车道。如果需要参与主线，可先在 CSV 中清除这些标记，让转换器不再跳过。
2. **补齐缺失车道段**：如果某些桩号区间缺行，需在 `Lane Link` 表中补全对应 lane number 的记录，保证每条车道在纵向上连续。
3. **必要时调整默认策略**：也可以在代码层面修改 `build_lane_topology` 的筛选逻辑，使特定的 `Lane Type` 或标记不被忽略，再重新运行转换。

只要 `Lane Link`/`Lane Division`/`Line Geometry` 三者描述保持一致，生成的 XODR 就会和原始车道配置对齐；否则就需要先从数据源着手修正。对于 US/JPN 两套配置而言，以上流程完全一致，只是默认宽度、路肩宽度等参数在配置文件里不同。

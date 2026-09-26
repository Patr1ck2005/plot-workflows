# Plot Workflows Agent Instructions

Navigation: [Repository homepage](README.md) · [Research-system map](D:/Obsidian/MyPhysics/System/Research-System-Map.md).

For work belonging to the user's research system, actually read the canonical
[MyPhysics system entry](D:/Obsidian/MyPhysics/System/README.md). Figure delivery
is owned by [Plot Foundation's style policy](D:/Dev/Projects/Work/plot-foundation/docs/style-policy.md#a4-paper-delivery);
this library's [style adaptation](docs/phase-04-style-profiles.md) only explains
how to consume it. Preserve the schema-neutral package boundary below.

## Boundary

This package owns reusable, schema-neutral orchestration for tracking,
visualization, batch composition, and multislice workflows. It depends on
`eigenmode-analysis` for numerical primitives and `plot-foundation` for
specifications, styles, and renderers.

Callers own file formats, column-name mappings, physical interpretation,
manifests, output paths, and provenance storage. Do not add simulator or
application-specific imports to this package.

## Data-space decision invariant

Before selecting a pipeline or Visualizer, classify the intrinsic data
dimension and sampling topology, then enumerate compatible views. Rendering
dimension does not determine data dimension: the same 2D scalar grid may be a
heatmap or a 3D surface. Follow the canonical
[Data Space and Visualization Views](docs/data-space-and-visualization.md)
specification and report its seven-field Agent decision record. If an analysis
operator reduces the raw object first, also report the raw space, operator,
derived coordinates/quantity, and validity or continuity rules. A peak ridge
or slice-wise extrema trajectory may be derived 1D data even when rendered as
a surface or 3D trajectory. Select a primary view and only useful
supplementary views; do not generate every candidate by default.

## Development

1. Characterize a workflow with synthetic inputs and caller-side tests.
2. Keep the reusable implementation here and add only thin caller adapters.
3. Run the package suite with optional extras needed by the changed module.
4. Record the installed shared-library versions in acceptance provenance.
5. Build a stable wheel before updating production callers.
6. Keep the canonical data-space specification and consumer Agent links intact.


<!-- project-hub:integration:start -->
## 项目总览接入约定

- 本项目的开发计划与验收证据继续保存在原有项目文档中，无需为总览维护额外进度摘要。总览直接读取本地 Git 提交；不再要求更新 `PROJECT_STATUS.md`，也不删除其他会话留下的状态原文。
- 若本项目已接入主要网页，入口以根目录 `PROJECT_WEB.json` 和 `WEB_ENTRY.md` 为准。修改网页入口、启动命令、依赖、端口或停止方式时，同步更新配置、统一启动脚本及入口说明，并验证启动、打开和停止。
- 使用 `start_web.bat`、`stop_web.bat`，需要加载后端或构建改动时使用 `start_web.bat restart`。开关由 Project Hub 统一管理；不要另起重复服务、按端口直接杀进程或自动换端口。
- Git 提交标题准确描述实际改动，不把提交活动视为完成度或验收证明。更新入口不构成提交授权；每次 commit 仍须向用户确认来源分类和具体模型，遵循既有 Origin 规则。
- 多会话修改前重新读取相关文件，只修改自己负责的内容；保留个人关注和备注。已有会话需重新读取本段，本约定不会自动同步会话或发送任务。
<!-- project-hub:integration:end -->

"""Independent CSV audit → candidate tracking → view specs → explicit exports.

Uses installed eigenmode-analysis 0.2.10, plot-foundation 0.1.13,
plot-workflows 0.1.7, and campaign-data 0.2.0 as a separate audit stage.
No sibling sources or production consumers are imported.
"""
from __future__ import annotations
import argparse
import contextlib
import io
from pathlib import Path
from dataclasses import replace
import numpy as np
from campaign_data import audit_directory, format_audit_report
from plot_workflows import CandidateSet, TrackingWorkflowOptions, run_path_workflow, band_plot_spec_from_payloads, surface_heatmap_spec_from_payload
from plot_workflows.multislice import SliceDataset, find_extrema_trajectory, inspect_slice_series, plot_heatmap_slices_3d
from plot_foundation import render_line_plot, render_heatmap, HeatmapStyle, SeriesStyle
from _showcase import COLORS, STYLE, Showcase, plt, sha, tidy3d


def audit_stage(out):
    data=out/"inputs";data.mkdir(exist_ok=True)
    path=data/"synthetic-scan.csv"
    path.write_text("x,y,value\n"+"".join(f"{i},{j},{i+2*j}\n" for i in range(4) for j in range(3) if (i,j)!=(2,1)),encoding="utf-8")
    before=sha(path)
    with contextlib.redirect_stdout(io.StringIO()):report=audit_directory(data,header_len=2,use_cache=False)
    assert report.files[0].actual_points==11 and report.files[0].expected_points==12
    assert sha(path)==before
    (out/"audit-report.txt").write_text(format_audit_report(report).replace(str(data),"inputs")+"\n",encoding="utf-8")
    return before


def tracked_payloads():
    t=np.linspace(-1,1,81);rng=np.random.default_rng(1729)
    frequencies=np.column_stack((1.2-np.sqrt((.28*t)**2+.07**2),1.2+np.sqrt((.28*t)**2+.07**2)))
    points=[]
    for i,row in enumerate(frequencies):
        order=rng.permutation(2);points.append(CandidateSet([f"{i}-{j}" for j in order],row[order,None]))
    workflow=run_path_workflow(points,t,options=TrackingWorkflowOptions(n_tracks=2,derivative_weights=.2))
    np.testing.assert_allclose(workflow.result.features[:,:,0],frequencies,atol=1e-12)
    assert workflow.coverage==1
    payloads={}
    for j in range(2):
        payloads[f"Branch {j+1}"]={"x":t,"y":{"frequency":workflow.result.features[:,j,0],"Q":(1+j)*1000/(.035+(t-.12*j)**2),"S3":(-1)**j*np.tanh(3*t)}}
    return payloads,workflow


def band_panel(ax,payloads,quantity,ylabel,profile="diagnostic",title=""):
    spec=band_plot_spec_from_payloads(payloads,title=title,ylabel=ylabel,quantity_name=quantity,is_tracked=True,xlabel="Parameter t",use_semilogy=quantity=="Q",style_profile=profile)
    # Explicit caller presentation choices, without changing any numerical payload.
    series=tuple(replace(s,style=SeriesStyle(color=COLORS[j],linewidth=.9)) for j,s in enumerate(spec.series))
    spec=replace(spec,series=series,axes=replace(spec.axes,grid=False,legend=profile=="diagnostic",legend_ncol=1),figure=replace(spec.figure,tight_layout=False))
    render_line_plot(spec,ax=ax)
    if profile=="diagnostic":
        ax.legend(frameon=False,fontsize=7,loc="center" if quantity=="frequency" else "upper right" if quantity=="Q" else "center left")
    ax.set_xlim(-1,1)
    if quantity=="S3":ax.set_ylim(-1,1)
    return spec


def slice_case():
    x,y=np.linspace(-1,1,43),np.linspace(-.7,.7,31);X,Y=np.meshgrid(x,y,indexing="ij")
    slices=[]
    for s in np.linspace(0,1,5):
        cx,cy=-.55+1.1*s,.24*np.sin(np.pi*s)
        amplitude=np.exp(-((X-cx)**2/.11+(Y-cy)**2/.075))
        slices.append(SliceDataset("s",float(s),{"x":x,"y":y},{"amplitude":amplitude}))
    report=inspect_slice_series(slices,x_key="x",y_key="y",field_keys="amplitude")
    assert report.common_shape==(43,31)
    trajectory=find_extrema_trajectory(slices,x_key="x",y_key="y",slice_key="s",field_key="amplitude",value_transform=None)
    return x,y,slices,trajectory


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source_hash=audit_stage(out)
    payloads,workflow=tracked_payloads()
    x,y,slices,trajectory=slice_case()
    trajectory.to_csv(out/"sampled-maxima.csv",index=False)
    show=Showcase(out,"Plot Workflows · synthetic end-to-end example",__file__,
        "All axes are dimensionless. The two synthetic branches are 1.2 ± sqrt((0.28t)²+0.07²). Q and S3 are independent illustrative channels, not inferred from the eigenvalues. Five Gaussian slices are sampled at s=0,0.25,0.5,0.75,1; there is no dense volume or interpolation between slices. Color on slices is amplitude in [0,1]. Maxima are discrete sample maxima, not mode identities. Workflow surface payloads are y-first and are transposed once by the adapter into foundation x-first specs.",
        [{"intrinsic_dimension":"1D","sampling_topology":"ordered path","parameter_space":"generic scalar parameter","quantity":"synthetic frequency, Q and S3","compatible_views":["bands","log lines","signed lines"],"selected_views":"common band/Q/polarization family","transformation_status":"candidate assignment preserves independent coordinate; Q and S3 are caller-supplied analytic channels"},
         {"intrinsic_dimension":"quasi-3D","sampling_topology":"five sparse 2D regular slices","parameter_space":"x,y plus slice parameter s","quantity":"Gaussian amplitude","compatible_views":["per-slice heatmap","stacked slices","extrema trajectory","projection"],"selected_views":"stacked slices, three representative heatmaps, sampled-maxima projection","transformation_status":"raw-plus-derived comparison; maxima reduce each slice to one sampled coordinate pair; no interpolation"}])
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,3,figsize=(210/25.4,73/25.4),layout="constrained")
        for ax,q,label,title in zip(axs,("frequency","Q","S3"),("Eigenvalue","Q (illustrative)","$S_3$ (illustrative)"),("Tracked branches","Positive log-scale channel","Signed polarization channel")):
            band_panel(ax,payloads,q,label,title=title)
        show.save(fig,"hero","A coherent band / Q / polarization family","All panels use the same tracked parameter path and branch colors; Q and S3 are independently defined example observables.")
        fig,axs=plt.subplots(1,2,figsize=(210/25.4,73/25.4),layout="constrained")
        a=band_panel(axs[0],payloads,"frequency","Eigenvalue",title="Diagnostic: labels and legend")
        b=band_panel(axs[1],payloads,"frequency","Eigenvalue",profile="publication_minimal")
        axs[1].set_title("Publication: minimal presentation",loc="left",fontsize=9)
        # Standalone gallery panels retain essential quantity labels as caller overrides.
        axs[1].set(xlabel="Parameter t",ylabel="Eigenvalue")
        for aa,bb in zip(a.series,b.series):np.testing.assert_array_equal(aa.y,bb.y)
        show.save(fig,"style-profiles","Same arrays, two presentation policies","The workflow maps diagnostic and publication_minimal profiles into specs. Essential labels are explicitly restored for this standalone gallery; numerical values are identical.")
        fig=plt.figure(figsize=(210/25.4,98/25.4));ax=fig.add_axes((.015,.07,.72,.87),projection="3d");cax=fig.add_axes((.82,.25,.02,.48))
        _,_,artists,mappable,mapping=plot_heatmap_slices_3d(slices,x_key="x",y_key="y",slice_key="s",field_key="amplitude",slice_direction="x",slice_position_mode="data",vmin=0,vmax=1,cmap="inferno",alpha=1,ax=ax,add_colorbar=False,view=(23,-54),box_aspect=(1.5,1.5,1))
        assert len(artists)==5
        tidy3d(ax)
        # Restore explicit slice locations after the generic 3D tick styling.
        ax.set_xticks([ds.slice_value for ds in slices])
        ax.set_xticklabels([f"{ds.slice_value:g}" for ds in slices])
        ax.set_yticks([-1,0,1]);ax.set_zticks([-.7,0,.7])
        cb=fig.colorbar(mappable,cax=cax);cb.set_label("Amplitude",fontsize=7);cb.ax.tick_params(labelsize=7)
        show.save(fig,"multislice","Five 2D slices in a quasi-3D view","Physical slice positions are retained. Each plane shows only its own grid, with a shared [0,1] color scale; no volume is inferred.")
        fig=plt.figure(figsize=(210/25.4,98/25.4),layout="constrained");gs=fig.add_gridspec(2,3,height_ratios=(1,1))
        for col,index in enumerate((0,2,4)):
            ds=slices[index];ax=fig.add_subplot(gs[0,col])
            # This workflow adapter explicitly consumes y-first values.
            payload={"x":x,"y":y,"values":ds.get_field("amplitude").T,"quantity_names":["Amplitude"]}
            spec=surface_heatmap_spec_from_payload(payload,title=f"s = {ds.slice_value:g}",cmap="inferno",contour=False)
            spec=replace(spec,style=HeatmapStyle(cmap="inferno",vmin=0,vmax=1),colorbar=False,axes=replace(spec.axes,xlabel="x",ylabel="y",grid=False),figure=replace(spec.figure,tight_layout=False))
            np.testing.assert_array_equal(spec.values,ds.get_field("amplitude"))
            render_heatmap(spec,ax=ax)
        ax=fig.add_subplot(gs[1,:]);ax.plot(trajectory["s"],trajectory["x"],"o-",color=COLORS[0],label="Sampled x at maximum",ms=4);ax.plot(trajectory["s"],trajectory["y"],"s-",color=COLORS[1],label="Sampled y at maximum",ms=4)
        ax.set(xlabel="Slice parameter s",ylabel="Peak coordinate",xticks=np.linspace(0,1,5));ax.legend(frameon=False,ncol=2,fontsize=7)
        show.save(fig,"slice-projection","Raw slices and derived sampled maxima","Three of five raw slices are shown above; all five sampled maxima are shown below. Connecting segments guide the eye and do not add sampled data.")
    expected_x=-.55+1.1*trajectory["s"].to_numpy();expected_y=.24*np.sin(np.pi*trajectory["s"].to_numpy())
    assert np.max(np.abs(trajectory["x"]-expected_x)) <= (x[1]-x[0])/2+1e-12
    assert np.max(np.abs(trajectory["y"]-expected_y)) <= (y[1]-y[0])/2+1e-12
    show.finish({"audit_source_sha256":source_hash,"audit_source_unchanged":True,"audited_points":"11/12","tracking_coverage":workflow.coverage,"tracking_expected_values_match":True,"profile_arrays_identical":True,"slice_count":len(slices),"slice_shape":[43,31],"maxima_within_half_grid_step":True,"payload_adapter_orientation_verified":True,"all_assertions_passed":True})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=Path("docs/assets/showcase"));main(p.parse_args().out)

<h1 align="center" style="margin-top: 0px;">Canny Edges</h1>
<h3 align="center" style="margin-top: 0px;">Find Edges via Canny Edge-Finding Algorithm: for Pytorch</h3>

<div id="img0" align="center">
    <img src="doc/images/img1.png" width="200" >
    <img src="doc/images/img1_out.png" width="200" >
    
</div>
<div id="img0" align="center">
    <img src="doc/images/img2.png" width="200" >
    <img src="doc/images/img2_out.png" width="200" >
</div>
<div id="img1" align="center">
    <img src="doc/images/img3.png" width="200" >
    <img src="doc/images/img3_out.png" width="200" >
</div>

&emsp;

A simple class to return a mask representing edges found by the Canny Edge-Finding algorithm. Supports images formatted in common ML batches [B, C, h, w], with float32 data, and values spanning [0,1]. 

Supports any number of channels C. Supports cpu and gpu data. Written entirely in native Pytorch-python. Supports TorchScript jit compilation (tested as of Pytorch 1.8.2)
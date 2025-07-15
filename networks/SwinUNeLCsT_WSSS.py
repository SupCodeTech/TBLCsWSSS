
# limitations under the License.

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock

from networks.SwinUNeLCsT_block import SwinUNeLCsTConvBlock, SwinUNeLCsTUpBlock, SwinUNeLCsTBlock
from tool.SGCSA_Module import SGCSA_Module

from monai.networks.blocks import Convolution
from networks.nest_transformer_3D import NestTransformer3D

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class GlobalMaxPool3d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool3d, self).__init__()
        self.gmp = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        x = self.gmp(x)
        return x.view(x.size(0), -1)  

class DXBlocks(nn.Module):
    def __init__(self, channels):
        super(CustomModule3D, self).__init__()

        self.depthwise_conv = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        self.layer_norm = nn.LayerNorm([channels, channels, channels, channels]) 
        
        self.expand_conv = nn.Conv3d(channels, channels * 4, kernel_size=1)
        
        self.squeeze_conv = nn.Conv3d(channels * 4, channels, kernel_size=1)
        
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        x = self.layer_norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)  # 调整维度以符合LayerNorm的输入要求
        
        x = self.expand_conv(x)
        
        x = self.gelu(x)
        
        x = self.squeeze_conv(x)
    
        x = 2 * (x + residual)
        
        return x

class SwinUNeLCsT(nn.Module):
    """
    SwinUNeLCsT model implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [128, 128, 96],
        feature_size: int = 16,
        patch_size: int = 4,
        depths: Tuple[int, int, int] = [2, 2, 8],
        num_heads: Tuple[int, int, int] = [4, 8, 16],
        embed_dim: Tuple[int, int, int] = [64, 128, 256, 512],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        self.sgsca = SGCSA_Module(num_layers=2, in_features=1, hidden_features=64)

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
    
        self.embed_dim = embed_dim

        self.nestViT = NestTransformer3D(
            img_size=96, 
            in_chans=1, 
            patch_size=patch_size, 
            num_levels=3, 
            embed_dims=embed_dim,                 
            num_heads=num_heads, 
            depths=depths, 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,                
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='', 
            weight_init='', 
            global_pool='avg',
        )

        self.encoder1 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = SwinUNeLCsTUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )
    
        self.encoder3 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],
            out_channels=feature_size * 32,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 16,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=32*feature_size,
            out_channels=64*feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_in):
        gmp = GlobalMaxPool3d()
        x, hidden_states_out = self.nestViT(x_in) 
        enc0 = self.encoder1(x_in) # 2, 32, 96, 96, 96 #SwinUNeLCsTConvBlock
        x1 = hidden_states_out[0] # 2, 128, 24, 24, 24
        enc1 = self.encoder2(x1) # 2, 64, 48, 48, 48  SwinUNeLCsTUpBlock
        x2 = hidden_states_out[1] # 2, 128, 24, 24, 24 
        enc2 = self.encoder3(x2) # 2, 128, 24, 24, 24 SwinUNeLCsTConvBlock
        x3 = hidden_states_out[2] # 2, 256, 12, 12, 12 
        enc3 = self.encoder4(x3) # 2, 256, 12, 12, 12 SwinUNeLCsTConvBlock
        x4 = hidden_states_out[3]
        enc4 = x4 # 2, 512, 6, 6, 6
        dec4 = x # 2, 512, 6, 6, 6
        dec4 = self.encoder10(dec4) 
        dec3 = self.decoder5(dec4, enc4) 
        dec3 = DXBlocks(dec3)
        dec2 = self.decoder4(dec3, enc3) 
        dec2 = DXBlocks(dec2)
        dec3 = gmp(dec3)
        dec2 = gmp(dec2)
        Location_output = torch.cat((dec3, dec2), dim=1)
        out = self.decoder1(dec0, enc0) 
        dec1 = self.decoder3(dec2, enc2) 
        dec1 = DXBlocks(dec1)
        dec0 = self.decoder2(dec1, enc1)
        dec0 = DXBlocks(dec0)
        dec1 = gmp(dec1)
        dec0 = gmp(dec0)
        Number_output = torch.cat((dec1, dec0), dim=1)
        out = self.decoder1(dec0, enc0) 
        logits = self.out(out)

        # 假设 hidden_states_out[-1] 是 Swin 输出的最后一层 MHSA 特征图，形状 [B, C, H, W, D]
        attn_feature = hidden_states_out[-1]  # [B, C, H, W, D]
        
        # Flatten spatial维度 [B, C, hwd]
        B, C, H, W, D = attn_feature.shape
        attn_feature_flat = attn_feature.view(B, C, -1).permute(0, 2, 1)  # [B, hwd, C]
        
        # 计算 pairwise similarity 矩阵，得到 Multi-head Attention Map (如 MHSA 中的 Attention Score)
        # [B, hwd, hwd, n_heads]  ——> 这里用点积模拟 (可根据实际情况替换为 Swin 的 Attention Map)
        similarity_list = []
        for b in range(B):
            feat = attn_feature_flat[b]  # [hwd, C]
            similarity = torch.einsum('ic,jc->ij', feat, feat)  # [hwd, hwd]
            similarity_list.append(similarity.unsqueeze(-1))
        attn_matrix = torch.stack(similarity_list, dim=0)  # [B, hwd, hwd, 1]
        
        # 送入 SGCSA 计算 W_aff
        W_aff_list = []
        for b in range(B):
            W_aff = self.sgsca(attn_matrix[b])  # [hwd, hwd]
            W_aff_list.append(W_aff)
        W_aff_batch = torch.stack(W_aff_list, dim=0)  # [B, hwd, hwd]
        
        return logits, Location_output, Number_output, W_aff_batch

class SwinUNeLCsT_ticv(nn.Module):
    """
    SwinUNeLCsT TICV model implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [128, 128, 96],
        feature_size: int = 16,
        patch_size: int = 2,
        depths: Tuple[int, int, int, int] = [2, 2, 2, 2],
        num_heads: Tuple[int, int, int, int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        
        self.embed_dim = [64, 128, 256, 512]

        self.nestViT = NestTransformer3D(
            img_size=96, 
            in_chans=1, 
            patch_size=4, 
            num_levels=3, 
            embed_dims=(64, 128, 256, 512),                 
            num_heads=(4, 8, 16), 
            depths=(2, 2, 8), 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,                
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='', 
            weight_init='', 
            global_pool='avg',
        )

        self.encoder1 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = SwinUNeLCsTUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )

        self.encoder3 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = SwinUNeLCsTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],
            out_channels=feature_size * 32,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 16,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = SwinUNeLCsTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=32*feature_size,
            out_channels=64*feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)  # type: ignore
        self.out_ticv = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=1)
        self.out_pfv = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=1)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
            
    def forward(self, x_in):
        x, hidden_states_out = self.nestViT(x_in) 
        enc0 = self.encoder1(x_in) # 2, 32, 96, 96, 96 #SwinUNeLCsTConvBlock
        x1 = hidden_states_out[0] # 2, 128, 24, 24, 24
        enc1 = self.encoder2(x1) # 2, 64, 48, 48, 48  SwinUNeLCsTUpBlock
        x2 = hidden_states_out[1] # 2, 128, 24, 24, 24 
        enc2 = self.encoder3(x2) # 2, 128, 24, 24, 24 SwinUNeLCsTConvBlock
        x3 = hidden_states_out[2] # 2, 256, 12, 12, 12 
        enc3 = self.encoder4(x3) # 2, 256, 12, 12, 12 SwinUNeLCsTConvBlock
        x4 = hidden_states_out[3]
        enc4 = x4 # 2, 512, 6, 6, 6
        dec4 = x # 2, 512, 6, 6, 6
        dec4 = self.encoder10(dec4) # 2, 1024, 3, 3, 3  Convolution
        dec3 = self.decoder5(dec4, enc4) # 2, 512, 6, 6, 6 SwinUNeLCsTBlock
        dec2 = self.decoder4(dec3, enc3) # 2, 256, 12, 12, 12
        dec1 = self.decoder3(dec2, enc2) # 2, 128, 24, 24, 24
        dec0 = self.decoder2(dec1, enc1) # 2, 64, 48, 48, 48
        out = self.decoder1(dec0, enc0) # 2, 32, 96, 96, 96
        logits = self.out(out)
        logits_ticv = self.out_ticv(out)
        logits_pfv = self.out_pfv(out)
        logits_out = torch.cat((logits, logits_ticv, logits_pfv), 1)
        return logits_out

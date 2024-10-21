import numpy as np
from typing import Tuple, Optional

class CFSM:
    def __init__(self, img_size: int = 112, style_dim: int = 128, n_bases: int = 10,
                 l_a: float = 0, u_a: float = 6, l_m: float = 0.05, u_m: float = 0.65):
        self.img_size = img_size
        self.style_dim = style_dim
        self.n_bases = n_bases
        self.l_a, self.u_a = l_a, u_a
        self.l_m, self.u_m = l_m, u_m
        
        self.encoder = ImageEncoder(img_size, style_dim)
        self.decoder = StyleDecoder(img_size, style_dim)
        self.discriminator = Discriminator(img_size)
        self.linear_subspace = LinearSubspace(n_bases, style_dim)
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        content = self.encoder(x)
        style = self.linear_subspace.sample()
        output = self.decoder(content, style)
        return output, style
    
    def compute_losses(self, real_img: np.ndarray, fake_img: np.ndarray, style: np.ndarray) -> dict:
        d_real = self.discriminator(real_img)
        d_fake = self.discriminator(fake_img)
        
        loss_d = -np.mean(np.log(d_real) + np.log(1 - d_fake))
        loss_g = -np.mean(np.log(d_fake))
        
        # Orthogonality loss
        loss_ort = np.abs(self.linear_subspace.compute_orthogonality_loss())
        
        # Identity preservation loss
        features_real = self.encoder.extract_features(real_img)
        features_fake = self.encoder.extract_features(fake_img)
        loss_id = self.compute_identity_loss(features_real, features_fake, style)
        
        return {
            'discriminator': loss_d,
            'generator': loss_g,
            'orthogonality': loss_ort,
            'identity': loss_id
        }
    
    def compute_identity_loss(self, f_real: np.ndarray, f_fake: np.ndarray, style: np.ndarray) -> float:
        cosine_sim = np.sum(f_real * f_fake) / (np.linalg.norm(f_real) * np.linalg.norm(f_fake))
        a = np.linalg.norm(style)
        g_a = self.l_m + (self.u_m - self.l_m) * (a - self.l_a) / (self.u_a - self.l_a)
        return np.square(1 - cosine_sim - g_a)

class ImageEncoder:
    def __init__(self, img_size: int, style_dim: int):
        self.conv1 = np.random.randn(16, 3, 3, 3) / np.sqrt(3 * 3 * 3)
        self.conv2 = np.random.randn(32, 16, 3, 3) / np.sqrt(16 * 3 * 3)
        self.conv3 = np.random.randn(64, 32, 3, 3) / np.sqrt(32 * 3 * 3)
        self.fc = np.random.randn(style_dim, 64 * (img_size // 8) * (img_size // 8)) / np.sqrt(64 * (img_size // 8) * (img_size // 8))
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = self._conv_block(x, self.conv1)
        h = self._conv_block(h, self.conv2)
        h = self._conv_block(h, self.conv3)
        h = h.reshape(h.shape[0], -1)
        return np.tanh(h @ self.fc.T)
    
    def _conv_block(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        h = self._conv2d(x, weight)
        h = np.maximum(0, h)  # ReLU
        return h[:, :, ::2, ::2]  # Simple stride-2 pooling
    
    def _conv2d(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        batch_size, in_c, h, w = x.shape
        out_c, _, kh, kw = weight.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        cols = np.zeros((batch_size, in_c * kh * kw, out_h * out_w))
        for y in range(out_h):
            for x in range(out_w):
                cols[:, :, y * out_w + x] = x[:, :, y:y + kh, x:x + kw].reshape(batch_size, -1)
        result = weight.reshape(out_c, -1) @ cols
        return result.reshape(batch_size, out_c, out_h, out_w)
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        return self(x)

class StyleDecoder:
    def __init__(self, img_size: int, style_dim: int):
        self.img_size = img_size
        self.style_dim = style_dim
        self.mlp = MLP(style_dim, style_dim)
        self.adain_params = np.random.randn(style_dim, style_dim * 2)
        
    def __call__(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        style_code = self.mlp(style)
        adain_params = style_code @ self.adain_params
        scaled_content = self._adaptive_instance_norm(content, adain_params)
        return self._decode(scaled_content)
    
    def _adaptive_instance_norm(self, content: np.ndarray, params: np.ndarray) -> np.ndarray:
        mean = np.mean(content, axis=(2, 3), keepdims=True)
        std = np.std(content, axis=(2, 3), keepdims=True) + 1e-8
        normalized = (content - mean) / std
        
        gamma = params[:, :self.style_dim].reshape(-1, self.style_dim, 1, 1)
        beta = params[:, self.style_dim:].reshape(-1, self.style_dim, 1, 1)
        
        return gamma * normalized + beta
    
    def _decode(self, h: np.ndarray) -> np.ndarray:
        # Simple upsampling decoder - in practice would be more sophisticated
        h = np.repeat(np.repeat(h, 2, axis=2), 2, axis=3)
        return np.tanh(h)

class Discriminator:
    def __init__(self, img_size: int):
        self.conv1 = np.random.randn(32, 3, 4, 4) / np.sqrt(3 * 4 * 4)
        self.conv2 = np.random.randn(64, 32, 4, 4) / np.sqrt(32 * 4 * 4)
        self.conv3 = np.random.randn(128, 64, 4, 4) / np.sqrt(64 * 4 * 4)
        self.fc = np.random.randn(1, 128 * (img_size // 8) * (img_size // 8)) / np.sqrt(128 * (img_size // 8) * (img_size // 8))
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = self._conv_block(x, self.conv1)
        h = self._conv_block(h, self.conv2)
        h = self._conv_block(h, self.conv3)
        h = h.reshape(h.shape[0], -1)
        return np.sigmoid(h @ self.fc.T)
    
    def _conv_block(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        h = self._conv2d(x, weight)
        h = np.maximum(0.2 * h, h)  # LeakyReLU
        return h[:, :, ::2, ::2]  # Stride-2 pooling

    def _conv2d(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return ImageEncoder._conv2d(self, x, weight)

class LinearSubspace:
    def __init__(self, n_bases: int, style_dim: int):
        self.bases = np.random.randn(n_bases, style_dim)
        self.mean_style = np.zeros(style_dim)
        
    def sample(self) -> np.ndarray:
        coef = np.random.normal(0, 1, size=len(self.bases))
        return coef @ self.bases + self.mean_style
    
    def compute_orthogonality_loss(self) -> float:
        normalized_bases = self.bases / np.linalg.norm(self.bases, axis=1, keepdims=True)
        gram_matrix = normalized_bases @ normalized_bases.T
        return np.sum(np.abs(gram_matrix - np.eye(len(self.bases))))

class MLP:
    def __init__(self, in_dim: int, out_dim: int):
        self.fc1 = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim)
        self.fc2 = np.random.randn(out_dim, out_dim) / np.sqrt(out_dim)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.fc1.T)
        return np.tanh(h @ self.fc2.T)

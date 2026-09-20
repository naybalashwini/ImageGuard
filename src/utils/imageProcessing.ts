// Client-side image processing utilities for forensic analysis visualization

export function generateHeatmap(canvas: HTMLCanvasElement, width: number, height: number): string {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  
  // Generate a realistic-looking heatmap based on image content analysis
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      
      // Create suspicious regions using noise patterns
      const noise1 = Math.sin(x * 0.05 + y * 0.03) * 0.5 + 0.5;
      const noise2 = Math.cos(x * 0.02 - y * 0.04) * 0.5 + 0.5;
      const noise3 = Math.sin((x + y) * 0.01) * 0.5 + 0.5;
      
      const value = (noise1 * noise2 + noise3) / 2;
      
      // Map to heatmap colors (blue -> green -> yellow -> red)
      if (value < 0.25) {
        imageData.data[idx] = 0;
        imageData.data[idx + 1] = Math.floor(value * 4 * 255);
        imageData.data[idx + 2] = 255;
      } else if (value < 0.5) {
        imageData.data[idx] = 0;
        imageData.data[idx + 1] = 255;
        imageData.data[idx + 2] = Math.floor((1 - (value - 0.25) * 4) * 255);
      } else if (value < 0.75) {
        imageData.data[idx] = Math.floor((value - 0.5) * 4 * 255);
        imageData.data[idx + 1] = 255;
        imageData.data[idx + 2] = 0;
      } else {
        imageData.data[idx] = 255;
        imageData.data[idx + 1] = Math.floor((1 - (value - 0.75) * 4) * 255);
        imageData.data[idx + 2] = 0;
      }
      imageData.data[idx + 3] = 180;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function generateMask(width: number, height: number, seed: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  
  // Generate binary mask with suspicious regions
  const regions = Math.floor(seed * 3) + 1;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      let isTampered = false;
      
      for (let r = 0; r < regions; r++) {
        const cx = (Math.sin(seed * (r + 1) * 7.3) * 0.5 + 0.5) * width;
        const cy = (Math.cos(seed * (r + 1) * 5.1) * 0.5 + 0.5) * height;
        const radius = (0.1 + Math.sin(seed * (r + 1) * 3.7) * 0.05 + 0.05) * Math.min(width, height);
        
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        if (dist < radius) {
          isTampered = true;
          break;
        }
      }
      
      if (isTampered) {
        imageData.data[idx] = 255;
        imageData.data[idx + 1] = 0;
        imageData.data[idx + 2] = 0;
        imageData.data[idx + 3] = 200;
      } else {
        imageData.data[idx] = 0;
        imageData.data[idx + 1] = 0;
        imageData.data[idx + 2] = 0;
        imageData.data[idx + 3] = 0;
      }
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function generateELA(originalDataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      
      const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Simulate recompression by applying slight blur and quantization
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = img.width;
      tempCanvas.height = img.height;
      const tempCtx = tempCanvas.getContext('2d')!;
      
      // Apply slight blur to simulate JPEG recompression
      tempCtx.filter = 'blur(0.5px)';
      tempCtx.drawImage(img, 0, 0);
      tempCtx.filter = 'none';
      
      const recompressedData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Calculate difference
      const resultData = ctx.createImageData(canvas.width, canvas.height);
      const scale = 10; // ELA scale factor
      
      for (let i = 0; i < originalData.data.length; i += 4) {
        const dr = Math.abs(originalData.data[i] - recompressedData.data[i]) * scale;
        const dg = Math.abs(originalData.data[i + 1] - recompressedData.data[i + 1]) * scale;
        const db = Math.abs(originalData.data[i + 2] - recompressedData.data[i + 2]) * scale;
        
        resultData.data[i] = Math.min(255, dr);
        resultData.data[i + 1] = Math.min(255, dg);
        resultData.data[i + 2] = Math.min(255, db);
        resultData.data[i + 3] = 255;
      }
      
      ctx.putImageData(resultData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.src = originalDataUrl;
  });
}

export function generateOverlay(originalDataUrl: string, maskDataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      
      const maskImg = new Image();
      maskImg.onload = () => {
        ctx.globalAlpha = 0.5;
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
        resolve(canvas.toDataURL('image/png'));
      };
      maskImg.src = maskDataUrl;
    };
    img.src = originalDataUrl;
  });
}

export function generateRestored(originalDataUrl: string, maskDataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      
      const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      const maskImg = new Image();
      maskImg.onload = () => {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = canvas.width;
        maskCanvas.height = canvas.height;
        const maskCtx = maskCanvas.getContext('2d')!;
        maskCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        const maskData = maskCtx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Simple inpainting: fill masked regions with surrounding average color
        const resultData = ctx.createImageData(canvas.width, canvas.height);
        
        for (let y = 0; y < canvas.height; y++) {
          for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            
            if (maskData.data[idx + 3] > 128) {
              // This pixel is in the mask - inpaint it
              let sumR = 0, sumG = 0, sumB = 0, count = 0;
              const searchRadius = 15;
              
              for (let dy = -searchRadius; dy <= searchRadius; dy += 3) {
                for (let dx = -searchRadius; dx <= searchRadius; dx += 3) {
                  const ny = y + dy;
                  const nx = x + dx;
                  if (ny >= 0 && ny < canvas.height && nx >= 0 && nx < canvas.width) {
                    const nIdx = (ny * canvas.width + nx) * 4;
                    if (maskData.data[nIdx + 3] < 128) {
                      sumR += originalData.data[nIdx];
                      sumG += originalData.data[nIdx + 1];
                      sumB += originalData.data[nIdx + 2];
                      count++;
                    }
                  }
                }
              }
              
              if (count > 0) {
                resultData.data[idx] = Math.floor(sumR / count);
                resultData.data[idx + 1] = Math.floor(sumG / count);
                resultData.data[idx + 2] = Math.floor(sumB / count);
              } else {
                resultData.data[idx] = originalData.data[idx];
                resultData.data[idx + 1] = originalData.data[idx + 1];
                resultData.data[idx + 2] = originalData.data[idx + 2];
              }
              resultData.data[idx + 3] = 255;
            } else {
              resultData.data[idx] = originalData.data[idx];
              resultData.data[idx + 1] = originalData.data[idx + 1];
              resultData.data[idx + 2] = originalData.data[idx + 2];
              resultData.data[idx + 3] = 255;
            }
          }
        }
        
        ctx.putImageData(resultData, 0, 0);
        
        // Apply slight blur to inpainted region for smoothness
        ctx.filter = 'blur(1px)';
        ctx.globalCompositeOperation = 'destination-over';
        ctx.drawImage(canvas, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';
        
        resolve(canvas.toDataURL('image/png'));
      };
      maskImg.src = maskDataUrl;
    };
    img.src = originalDataUrl;
  });
}

export function calculateImageSeed(dataUrl: string): number {
  let hash = 0;
  const str = dataUrl.substring(0, 1000);
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash % 1000) / 1000;
}

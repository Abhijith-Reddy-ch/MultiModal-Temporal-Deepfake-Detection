const API_BASE_URL = '/api';

async function parseErrorBody(response) {
  try {
    const data = await response.json();
    return data.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function getHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
    if (!response.ok) return { status: 'error', model_loaded: false };
    return response.json();
  } catch {
    return { status: 'error', model_loaded: false };
  }
}

export async function predict(file) {
  const formData = new FormData();
  formData.append('file', file, file.name || 'upload.mp4');

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorBody(response));
  }
  return response.json();
}

export async function getGradcam(file) {
  const formData = new FormData();
  formData.append('file', file, file.name || 'upload.mp4');

  const response = await fetch(`${API_BASE_URL}/gradcam`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorBody(response));
  }
  return response.json();
}

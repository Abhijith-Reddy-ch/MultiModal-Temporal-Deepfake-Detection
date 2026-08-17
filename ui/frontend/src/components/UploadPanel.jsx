'use client';

import { useCallback, useRef, useState } from 'react';
import { UploadCloud, FileVideo, X } from 'lucide-react';
import clsx from 'clsx';

const ACCEPTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

export default function UploadPanel({ file, onFileSelected, onPredict, onExplain, busy }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (fileList) => {
      const picked = fileList?.[0];
      if (!picked) return;
      onFileSelected(picked);
    },
    [onFileSelected]
  );

  const previewUrl = file ? URL.createObjectURL(file) : null;

  return (
    <div className="w-full">
      <div
        className={clsx(
          'flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors',
          dragOver
            ? 'border-series-1 bg-series-1/5'
            : 'border-line-grid dark:border-line-grid-dark'
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        {!file ? (
          <>
            <UploadCloud className="h-8 w-8 text-ink-muted" aria-hidden="true" />
            <p className="text-sm text-ink-secondary dark:text-ink-secondary-dark">
              Drag a video here, or{' '}
              <button
                type="button"
                className="font-medium text-series-1 underline underline-offset-2"
                onClick={() => inputRef.current?.click()}
              >
                browse files
              </button>
            </p>
            <p className="text-xs text-ink-muted">{ACCEPTED_EXTENSIONS.join(', ')}</p>
          </>
        ) : (
          <div className="flex w-full max-w-md items-center gap-3 rounded-lg border border-line-grid bg-surface p-3 text-left dark:border-line-grid-dark dark:bg-surface-dark">
            <FileVideo className="h-6 w-6 shrink-0 text-series-1" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="text-xs text-ink-muted">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
            <button
              type="button"
              aria-label="Remove file"
              className="rounded p-1 text-ink-muted hover:bg-line-grid dark:hover:bg-line-grid-dark"
              onClick={() => onFileSelected(null)}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {previewUrl && (
        <video
          key={previewUrl}
          src={previewUrl}
          controls
          className="mt-4 max-h-64 w-full rounded-lg border border-line-grid dark:border-line-grid-dark"
        />
      )}

      <div className="mt-4 flex gap-3">
        <button
          type="button"
          disabled={!file || busy}
          onClick={onPredict}
          className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40 hover:opacity-90"
        >
          {busy === 'predict' ? 'Analyzing…' : 'Run Detection'}
        </button>
        <button
          type="button"
          disabled={!file || busy}
          onClick={onExplain}
          className="rounded-lg border border-line-grid px-4 py-2 text-sm font-medium text-ink-primary transition-colors disabled:cursor-not-allowed disabled:opacity-40 hover:bg-line-grid/40 dark:border-line-grid-dark dark:text-ink-primary-dark dark:hover:bg-line-grid-dark/40"
        >
          {busy === 'explain' ? 'Explaining…' : 'Explain (Grad-CAM)'}
        </button>
      </div>
    </div>
  );
}

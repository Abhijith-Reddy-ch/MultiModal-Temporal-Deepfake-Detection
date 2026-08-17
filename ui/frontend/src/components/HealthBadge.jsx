'use client';

import { useEffect, useState } from 'react';
import { getHealth } from '@/lib/api';

export default function HealthBadge() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      const result = await getHealth();
      if (!cancelled) setHealth(result);
    }
    check();
    const id = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const ok = health?.status === 'ok' && health?.model_loaded;
  const label = health === null ? 'Checking backend…' : ok ? 'Model online' : 'Backend unavailable';
  const dotColor = health === null ? 'bg-ink-muted' : ok ? 'bg-status-good' : 'bg-status-critical';

  return (
    <div className="flex items-center gap-2 text-sm text-ink-secondary dark:text-ink-secondary-dark">
      <span className={`h-2 w-2 rounded-full ${dotColor}`} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

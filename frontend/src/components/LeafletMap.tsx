import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Waypoint } from '../types';

interface LeafletMapProps {
  waypoints: Waypoint[];
  highlightAnomalies?: boolean;
}

export const LeafletMap: React.FC<LeafletMapProps> = ({ waypoints, highlightAnomalies = true }) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  // Initialize Map once
  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Safety clear in case of StrictMode remount
    const container = mapContainerRef.current as HTMLElement & { _leaflet_id?: unknown };
    if (container._leaflet_id) {
      delete container._leaflet_id;
    }

    const map = L.map(container, {
      center: [48.85, 12.0],
      zoom: 4,
      zoomControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 18,
    }).addTo(map);

    const layers = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
    layerGroupRef.current = layers;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      layerGroupRef.current = null;
    };
  }, []);

  // Update Layers & Bounds when waypoints change
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layers = layerGroupRef.current;
    if (!map || !layers) return;

    layers.clearLayers();

    if (waypoints.length >= 2) {
      const latlngs: L.LatLngExpression[] = waypoints.map(w => [w.latitude, w.longitude]);
      
      // Polyline for full route
      const routeLine = L.polyline(latlngs, {
        color: '#2563eb',
        weight: 3.5,
        opacity: 0.85,
      }).addTo(layers);

      // Origin Marker
      const origin = waypoints[0];
      L.circleMarker([origin.latitude, origin.longitude], {
        radius: 6,
        color: '#16a34a',
        fillColor: '#22c55e',
        fillOpacity: 1,
      })
        .bindPopup(`<b>Origin</b><br>FL${origin.flight_level}<br>${origin.time_over}`)
        .addTo(layers);

      // Destination Marker
      const dest = waypoints[waypoints.length - 1];
      L.circleMarker([dest.latitude, dest.longitude], {
        radius: 6,
        color: '#dc2626',
        fillColor: '#ef4444',
        fillOpacity: 1,
      })
        .bindPopup(`<b>Destination</b><br>FL${dest.flight_level}<br>${dest.time_over}`)
        .addTo(layers);

      // Waypoint markers & anomaly segments
      waypoints.slice(1, -1).forEach(w => {
        const isAnom = highlightAnomalies && w.is_anomaly;
        const marker = L.circleMarker([w.latitude, w.longitude], {
          radius: isAnom ? 7 : 4,
          color: isAnom ? '#b91c1c' : '#3b82f6',
          fillColor: isAnom ? '#f87171' : '#60a5fa',
          fillOpacity: 0.9,
          weight: isAnom ? 2.5 : 1,
        });

        marker.bindPopup(`
          <div style="font-family: sans-serif; font-size: 12px;">
            <b>Waypoint #${w.seq_num}</b><br/>
            Time: ${w.time_over}<br/>
            Alt: FL${w.flight_level} (${w.flight_level * 100} ft)<br/>
            Speed: ${w.speed_kts ? `${Math.round(w.speed_kts)} kts` : 'N/A'}<br/>
            ${isAnom ? `<span style="color: red; font-weight: bold;">⚠️ ${w.anomaly_type || 'Anomaly'}: ${w.description || ''}</span>` : ''}
          </div>
        `);
        marker.addTo(layers);
      });

      // Fit bounds
      map.fitBounds(routeLine.getBounds(), { padding: [30, 30] });
    }
  }, [waypoints, highlightAnomalies]);

  return (
    <div
      ref={mapContainerRef}
      style={{
        width: '100%',
        height: '100%',
        minHeight: '380px',
        borderRadius: '8px',
        zIndex: 1,
      }}
    />
  );
};

declare module 'react-simple-maps' {
  import { ReactNode, CSSProperties } from 'react';

  interface ProjectionConfig {
    center?: [number, number];
    scale?: number;
    rotate?: [number, number, number];
  }

  interface ComposableMapProps {
    projection?: string;
    projectionConfig?: ProjectionConfig;
    width?: number;
    height?: number;
    style?: CSSProperties;
    children?: ReactNode;
  }

  interface GeographiesProps {
    geography: string | object;
    children: (data: { geographies: any[] }) => ReactNode;
  }

  interface GeographyProps {
    geography: any;
    key?: string;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
    style?: {
      default?: CSSProperties;
      hover?: CSSProperties;
      pressed?: CSSProperties;
    };
  }

  interface LineProps {
    from: [number, number];
    to: [number, number];
    stroke?: string;
    strokeWidth?: number;
    strokeLinecap?: string;
    strokeDasharray?: string;
  }

  interface MarkerProps {
    coordinates: [number, number];
    children?: ReactNode;
  }

  export const ComposableMap: React.FC<ComposableMapProps>;
  export const Geographies: React.FC<GeographiesProps>;
  export const Geography: React.FC<GeographyProps>;
  export const Line: React.FC<LineProps>;
  export const Marker: React.FC<MarkerProps>;
}

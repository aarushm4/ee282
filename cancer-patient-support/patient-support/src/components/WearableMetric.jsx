import { TrendIndicator } from "./TrendIndicator";

export const WearableMetric = ({ icon, name, value, trend }) => (
  <div className='flex items-center'>
    {icon}
    <div className='ml-2'>
      <p className='text-sm font-semibold'>{name}</p>
      <p className='text-lg'>{value}</p>
      <TrendIndicator trend={trend} />
    </div>
  </div>
);
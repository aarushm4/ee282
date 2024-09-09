import { Activity, AlertCircle, Heart, Moon } from "lucide-react";
import { WearableMetric } from "./WearableMetric";

export const WearableDataAnalysis = ({ wearableData }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Wearable Data Analysis</h2>
    <div className='grid grid-cols-2 gap-4'>
      <WearableMetric icon={<Heart className='w-6 h-6 text-red-500' />} name='Heart Rate' value={wearableData.latest.heart_rate} trend={wearableData.trends.heart_rate} />
      <WearableMetric icon={<Activity className='w-6 h-6 text-green-500' />} name='Steps' value={wearableData.latest.steps} trend={wearableData.trends.steps} />
      <WearableMetric icon={<Moon className='w-6 h-6 text-blue-500' />} name='Sleep Hours' value={wearableData.latest.sleep_hours.toFixed(1)} trend={wearableData.trends.sleep_hours} />
      <WearableMetric icon={<AlertCircle className='w-6 h-6 text-yellow-500' />} name='Stress Level' value={wearableData.latest.stress_level} trend={wearableData.trends.stress_level} />
    </div>
  </div>
);
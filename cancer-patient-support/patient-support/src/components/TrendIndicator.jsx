import { TrendingUp } from "lucide-react";

export const TrendIndicator = ({ trend }) => {
  const getColor = () => {
    switch (trend) {
      case 'increasing':
        return 'text-green-500';
      case 'decreasing':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className={`flex items-center ${getColor()}`}>
      <TrendingUp className='w-4 h-4 mr-1' />
      <span className='text-sm capitalize'>{trend}</span>
    </div>
  );
};
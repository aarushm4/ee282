export const FeedbackAnalysis = ({ feedback, sentiment }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Feedback Analysis</h2>
    <p className='mb-2'>
      <strong>Recent Feedback:</strong> {feedback}
    </p>
    <p>
      <strong>Sentiment:</strong> {sentiment.positive_sentiment_probability > 0.5 ? 'Positive' : 'Negative'} ({(sentiment.positive_sentiment_probability * 100).toFixed(1)}% positive)
    </p>
  </div>
);

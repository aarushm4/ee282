import axios from "axios";
import { useEffect, useState } from "react";
import { API_BASE_URL } from "../constants";
import { PatientInfo } from "./PatientInfo";
import { OutcomePrediction } from "./OutcomePrediction";
import { ClinicalTrials } from "./ClinicalTrials";
import { TreatmentRecommendations } from "./TreatmentRecommandations";
import { WearableDataAnalysis } from "./WearableDataAnalysis";
import { FeedbackAnalysis } from "./FeedbackAnalysis";

export const Dashboard = ({ patientId }) => {
  const [dashboardData, setDashboardData] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/patient-dashboard?patient_id=${patientId}`);
        setDashboardData(response.data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      }
    };
    if (patientId > 0) {
      fetchDashboardData();
    }
  }, [patientId]);

  if (!dashboardData) return <div className='text-center'>Loading...</div>;

  return (
    <div className='container mx-auto px-4 py-8'>
      <h1 className='text-3xl font-bold mb-6'>Patient Dashboard</h1>

      <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
        <PatientInfo patientInfo={dashboardData.patient_info} />
        <OutcomePrediction prediction={dashboardData.outcome_prediction} />
        <ClinicalTrials trials={dashboardData.clinical_trials} />
        <TreatmentRecommendations recommendations={dashboardData.treatment_recommendations} />
        <WearableDataAnalysis wearableData={dashboardData.wearable_data_analysis} />
        <FeedbackAnalysis feedback={dashboardData.recent_feedback} sentiment={dashboardData.sentiment_analysis} />
      </div>
    </div>
  );
};
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import matplotlib.pyplot as plt
import io
import base64
import os
from datetime import datetime

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                alignment=TA_CENTER,
                spaceAfter=30
            ),
            'heading': ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=colors.darkgreen,
                spaceAfter=12
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=self.styles['Normal'],
                fontSize=11,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            )
        }

    def create_emotion_spectrum_chart(self, emotion_data):
        """Create emotion spectrum visualization supporting intelligent emotion analysis."""
        plt.figure(figsize=(12, 8))
        
        analysis_type = emotion_data.get('analysis_type', 'unknown')
        
        if analysis_type == 'face_enhanced_with_scene':
            # Face analysis enhanced with scene context
            emotion_dist = emotion_data.get('emotion_distribution', {})
            spectrum_metrics = emotion_data.get('emotion_spectrum', {})
            scene_context = emotion_data.get('scene_context', {})
            
            # Face emotions pie chart
            if emotion_dist:
                plt.subplot(2, 2, 1)
                emotions = list(emotion_dist.keys())
                values = list(emotion_dist.values())
                colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
                plt.pie(values, labels=emotions, autopct='%1.1f%%', colors=colors_map)
                plt.title('Facial Emotion Distribution')
            
            # Scene context
            plt.subplot(2, 2, 2)
            context_data = [
                spectrum_metrics.get('positivity_score', 0),
                spectrum_metrics.get('combined_positivity', 0),
                scene_context.get('scene_influence', 0)
            ]
            context_labels = ['Face Positivity', 'Combined Positivity', 'Scene Influence']
            plt.bar(context_labels, context_data, color=['#3498DB', '#2ECC71', '#E67E22'])
            plt.title('Face + Scene Context Analysis')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
        elif analysis_type == 'scene_based':
            # Scene-only analysis
            spectrum_metrics = emotion_data.get('emotion_spectrum', {})
            scene_details = emotion_data.get('scene_details', {})
            
            # Scene emotions
            plt.subplot(2, 2, 1)
            scene_emotions = spectrum_metrics.get('scene_emotions', [])
            if scene_emotions:
                plt.bar(range(len(scene_emotions)), [1] * len(scene_emotions), 
                       color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'][:len(scene_emotions)])
                plt.xticks(range(len(scene_emotions)), scene_emotions, rotation=45)
                plt.title('Scene Emotions Detected')
                plt.ylabel('Presence')
            
            # Scene features
            plt.subplot(2, 2, 2)
            features = scene_details.get('scene_features', {})
            feature_names = ['Brightness', 'Saturation', 'Warmth', 'Contrast']
            feature_values = [
                features.get('brightness', 0),
                features.get('saturation', 0),
                features.get('warmth_ratio', 0),
                features.get('contrast', 0)
            ]
            plt.bar(feature_names, feature_values, color=['#F1C40F', '#E67E22', '#E74C3C', '#34495E'])
            plt.title('Scene Visual Features')
            plt.ylabel('Score/Ratio')
            plt.ylim(0, 1)
            
        else:
            # Traditional face-only or fallback analysis
            emotion_dist = emotion_data.get('emotion_distribution', {})
            spectrum_metrics = emotion_data.get('emotion_spectrum', {})
            
            if emotion_dist:
                plt.subplot(2, 2, 1)
                emotions = list(emotion_dist.keys())
                values = list(emotion_dist.values())
                colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
                plt.pie(values, labels=emotions, autopct='%1.1f%%', colors=colors_map)
                plt.title('Emotion Distribution')
        
        # Common spectrum analysis (works for all types)
        if 'emotion_spectrum' in emotion_data:
            plt.subplot(2, 2, 4)
            spectrum_metrics = emotion_data['emotion_spectrum']
            spectrum_labels = ['Positivity', 'Negativity', 'Enthusiasm', 'Intensity']
            spectrum_values = [
                spectrum_metrics.get('positivity_score', 0),
                spectrum_metrics.get('negativity_score', 0),
                spectrum_metrics.get('enthusiasm_score', 0),
                spectrum_metrics.get('emotional_intensity', 0)
            ]
            
            plt.bar(spectrum_labels, spectrum_values, color=['#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'])
            plt.title('Emotion Spectrum Analysis')
            plt.ylabel('Score')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer

    def create_composition_chart(self, composition_data):
        """Create composition analysis visualization."""
        metrics = ['Symmetry', 'Balance', 'Rule of Thirds', 'Overall Aesthetic']
        scores = [
            composition_data.get('symmetry_score', 0),
            composition_data.get('balance_score', 0),
            composition_data.get('rule_of_thirds_score', 0),
            composition_data.get('overall_aesthetic_score', 0)
        ]
        
        plt.figure(figsize=(10, 6))
        
        # Radar chart
        angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
        angles += angles[:1]
        scores += scores[:1]
        
        plt.subplot(polar=True)
        plt.plot(angles, scores, 'o-', linewidth=2, color='#3498DB')
        plt.fill(angles, scores, alpha=0.25, color='#3498DB')
        plt.xticks(angles[:-1], metrics)
        plt.ylim(0, 1)
        plt.title('Composition Analysis', pad=20)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer

    def create_lifestyle_chart(self, object_data):
        """Create lifestyle analysis visualization."""
        categories = object_data.get('lifestyle_categories', {})
        if not categories:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Horizontal bar chart
        category_names = list(categories.keys())
        category_counts = list(categories.values())
        
        plt.barh(category_names, category_counts, color='#E67E22')
        plt.title('Lifestyle Categories Detected')
        plt.xlabel('Count')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer

    def generate_summary_scores(self, analysis_results):
        """Generate overall summary scores."""
        scores = {}
        
        # Average scores across all images
        total_images = len(analysis_results)
        if total_images == 0:
            return scores
        
        # Emotion scores
        emotion_scores = {
            'avg_positivity': 0,
            'avg_enthusiasm': 0,
            'social_preference': 'unknown'
        }
        
        # Composition scores
        composition_scores = {
            'avg_aesthetic': 0,
            'avg_symmetry': 0,
            'avg_balance': 0
        }
        
        # Lifestyle scores
        lifestyle_scores = {
            'activity_level': 0,
            'sophistication': 0,
            'diversity': 0
        }
        
        solo_count = 0
        for result in analysis_results:
            # Emotion metrics
            emotion_data = result.get('facial_emotion', {})
            spectrum = emotion_data.get('emotion_spectrum', {})
            emotion_scores['avg_positivity'] += spectrum.get('positivity_score', 0)
            emotion_scores['avg_enthusiasm'] += spectrum.get('enthusiasm_score', 0)
            
            social_dynamics = emotion_data.get('social_dynamics', {})
            if social_dynamics.get('social_tendency') == 'solo':
                solo_count += 1
            
            # Composition metrics
            comp_data = result.get('composition', {})
            composition_scores['avg_aesthetic'] += comp_data.get('overall_aesthetic_score', 0)
            composition_scores['avg_symmetry'] += comp_data.get('symmetry_score', 0)
            composition_scores['avg_balance'] += comp_data.get('balance_score', 0)
            
            # Lifestyle metrics
            obj_data = result.get('object_recognition', {})
            lifestyle_scores['activity_level'] += obj_data.get('activity_level', 0)
            lifestyle_scores['sophistication'] += obj_data.get('sophistication_score', 0)
        
        # Calculate averages
        for key in emotion_scores:
            if key != 'social_preference':
                emotion_scores[key] = round(emotion_scores[key] / total_images, 3)
        
        emotion_scores['social_preference'] = 'solo' if solo_count > total_images / 2 else 'social'
        
        for key in composition_scores:
            composition_scores[key] = round(composition_scores[key] / total_images, 3)
        
        for key in lifestyle_scores:
            lifestyle_scores[key] = round(lifestyle_scores[key] / total_images, 3)
        
        return {
            'emotion': emotion_scores,
            'composition': composition_scores,
            'lifestyle': lifestyle_scores
        }

    def generate_pdf_report(self, analysis_results, output_path="data/visual_persona_report.pdf"):
        """Generate comprehensive PDF report."""
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=inch)
        story = []
        
        # Title
        title = Paragraph("Visual Persona Analysis Report", self.custom_styles['title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Timestamp
        timestamp = Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                            self.custom_styles['body'])
        story.append(timestamp)
        story.append(Spacer(1, 24))
        
        # Executive Summary
        summary_scores = self.generate_summary_scores(analysis_results)
        
        story.append(Paragraph("Executive Summary", self.custom_styles['heading']))
        
        summary_text = f"""
        This report analyzes {len(analysis_results)} images to generate a comprehensive visual personality profile.
        The analysis covers emotional expression, aesthetic composition, lifestyle patterns, and social behavior.
        
        <b>Key Findings:</b><br/>
        • Average Positivity Score: {summary_scores.get('emotion', {}).get('avg_positivity', 0):.2f}/1.00<br/>
        • Average Enthusiasm Level: {summary_scores.get('emotion', {}).get('avg_enthusiasm', 0):.2f}/1.00<br/>
        • Aesthetic Quality Score: {summary_scores.get('composition', {}).get('avg_aesthetic', 0):.2f}/1.00<br/>
        • Social Preference: {summary_scores.get('emotion', {}).get('social_preference', 'Unknown').title()}<br/>
        • Activity Level: {summary_scores.get('lifestyle', {}).get('activity_level', 0):.1f}
        """
        
        story.append(Paragraph(summary_text, self.custom_styles['body']))
        story.append(Spacer(1, 24))
        
        # Detailed Analysis for each image
        for i, result in enumerate(analysis_results, 1):
            story.append(Paragraph(f"Image {i} Analysis", self.custom_styles['heading']))
            
            # Emotion Analysis
            emotion_data = result.get('facial_emotion', {})
            if emotion_data and 'error' not in emotion_data:
                story.append(Paragraph("Emotional Expression", self.custom_styles['heading']))
                
                # Create emotion chart
                emotion_chart = self.create_emotion_spectrum_chart(emotion_data)
                img = Image(emotion_chart, width=6*inch, height=3.6*inch)
                story.append(img)
                story.append(Spacer(1, 12))
                
                # Emotion details
                spectrum = emotion_data.get('emotion_spectrum', {})
                social_dynamics = emotion_data.get('social_dynamics', {})
                
                emotion_text = f"""
                <b>Dominant Emotion:</b> {spectrum.get('dominant_emotion', 'Unknown').title()}<br/>
                <b>Positivity Score:</b> {spectrum.get('positivity_score', 0):.3f}<br/>
                <b>Enthusiasm Level:</b> {spectrum.get('enthusiasm_score', 0):.3f}<br/>
                <b>Emotional Intensity:</b> {spectrum.get('emotional_intensity', 0):.3f}<br/>
                <b>Social Tendency:</b> {social_dynamics.get('social_tendency', 'Unknown').replace('_', ' ').title()}
                """
                
                story.append(Paragraph(emotion_text, self.custom_styles['body']))
                story.append(Spacer(1, 12))
            
            # Composition Analysis
            comp_data = result.get('composition', {})
            if comp_data:
                story.append(Paragraph("Aesthetic Composition", self.custom_styles['heading']))
                
                comp_chart = self.create_composition_chart(comp_data)
                img = Image(comp_chart, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            
            # Lifestyle Analysis
            obj_data = result.get('object_recognition', {})
            if obj_data and obj_data.get('lifestyle_categories'):
                story.append(Paragraph("Lifestyle Patterns", self.custom_styles['heading']))
                
                lifestyle_chart = self.create_lifestyle_chart(obj_data)
                if lifestyle_chart:
                    img = Image(lifestyle_chart, width=5*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                
                lifestyle_text = f"""
                <b>Detected Objects:</b> {', '.join(obj_data.get('detected_objects', [])[:10])}<br/>
                <b>Activity Level:</b> {obj_data.get('activity_level', 0)}<br/>
                <b>Sophistication Score:</b> {obj_data.get('sophistication_score', 0)}
                """
                
                story.append(Paragraph(lifestyle_text, self.custom_styles['body']))
            
            # Color Analysis
            color_data = result.get('color_analysis', {})
            if color_data:
                story.append(Paragraph("Color Psychology", self.custom_styles['heading']))
                
                color_text = f"""
                <b>Dominant Mood:</b> {color_data.get('mood', 'Unknown').title()}<br/>
                <b>Personality Traits:</b> {', '.join(color_data.get('personality_traits', []))}<br/>
                """
                
                story.append(Paragraph(color_text, self.custom_styles['body']))
            
            if i < len(analysis_results):
                story.append(PageBreak())
            else:
                story.append(Spacer(1, 24))
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.custom_styles['heading']))
        
        conclusion_text = """
        This visual persona analysis provides insights into emotional expression patterns, aesthetic preferences, 
        and lifestyle indicators based on image content. The analysis should be considered alongside other 
        factors for a complete personality assessment.
        
        For questions about this report or the analysis methodology, please refer to the project documentation.
        """
        
        story.append(Paragraph(conclusion_text, self.custom_styles['body']))
        
        # Build PDF
        doc.build(story)
        return output_path
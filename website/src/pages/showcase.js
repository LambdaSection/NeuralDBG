import React, { useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const showcaseProjects = [
  {
    title: 'Medical Image Classification',
    description: 'CNN-based system for detecting diseases from X-ray images with 95% accuracy. Deployed in 15 hospitals across North America, processing 10,000+ images daily.',
    author: 'Dr. Sarah Chen',
    organization: 'Stanford Medical AI Lab',
    image: '/img/showcase/medical.jpg',
    tags: ['Healthcare', 'CNN', 'Production'],
    githubUrl: 'https://github.com/example/medical-ai',
    demoUrl: 'https://demo.example.com',
    caseStudyUrl: '/blog/case-study-medical-imaging',
    impact: 'Reduced diagnosis time by 60%, helping 50,000+ patients',
    featured: true,
    successStory: 'Saved an estimated $2M in diagnostic costs and improved early detection rates by 40%',
    metrics: { accuracy: '95%', throughput: '10K/day', uptime: '99.9%' },
  },
  {
    title: 'Real-time Sentiment Analysis',
    description: 'LSTM model analyzing customer feedback in real-time for Fortune 500 company. Processing 1M+ reviews monthly across 20 languages.',
    author: 'Alex Rodriguez',
    organization: 'TechCorp Analytics',
    image: '/img/showcase/sentiment.jpg',
    tags: ['NLP', 'LSTM', 'Enterprise'],
    githubUrl: 'https://github.com/example/sentiment-analysis',
    caseStudyUrl: '/blog/case-study-sentiment-analysis',
    impact: 'Improved customer satisfaction by 35%',
    featured: true,
    successStory: 'Helped identify and resolve 2,000+ critical issues before escalation',
    metrics: { accuracy: '92%', latency: '50ms', languages: '20' },
  },
  {
    title: 'Autonomous Vehicle Vision',
    description: 'Object detection and tracking system for self-driving cars. Powers Level 4 autonomous vehicles in urban environments.',
    author: 'Michael Zhang',
    organization: 'AutoDrive Labs',
    image: '/img/showcase/autonomous.jpg',
    tags: ['Computer Vision', 'Real-time', 'Production'],
    githubUrl: 'https://github.com/example/autodrive-vision',
    paperUrl: 'https://arxiv.org/example',
    impact: 'Enabled safe autonomous driving in 5 cities',
    featured: true,
    successStory: 'Zero accidents in 500,000+ autonomous miles driven',
    metrics: { fps: '60', latency: '10ms', accuracy: '99.8%' },
  },
  {
    title: 'Financial Fraud Detection',
    description: 'Neural network detecting fraudulent transactions with 99.2% precision. Protecting $500M+ in daily transactions.',
    author: 'Emily Johnson',
    organization: 'SecureBank AI',
    image: '/img/showcase/fraud.jpg',
    tags: ['Finance', 'Anomaly Detection', 'Production'],
    githubUrl: 'https://github.com/example/fraud-detection',
    caseStudyUrl: '/blog/case-study-fraud-detection',
    impact: 'Prevented $50M in fraudulent transactions',
    successStory: 'Reduced false positives by 70% while catching 99.2% of fraud',
    metrics: { precision: '99.2%', recall: '98.5%', volume: '$500M/day' },
  },
  {
    title: 'Speech Recognition System',
    description: 'Multi-language speech-to-text with transformer architecture. Supporting 50+ languages with real-time transcription.',
    author: 'David Park',
    organization: 'VoiceAI Research',
    image: '/img/showcase/speech.jpg',
    tags: ['Audio', 'Transformer', 'Research'],
    githubUrl: 'https://github.com/example/speech-recognition',
    paperUrl: 'https://arxiv.org/example',
    impact: 'Enabling communication for 1M+ users globally',
    metrics: { wer: '5.2%', languages: '50+', latency: '100ms' },
  },
  {
    title: 'E-commerce Recommendation Engine',
    description: 'Personalized product recommendations serving 10M+ users daily. Increased conversion rates by 45%.',
    author: 'Lisa Wang',
    organization: 'ShopMart',
    image: '/img/showcase/ecommerce.jpg',
    tags: ['Recommender Systems', 'Production', 'Scale'],
    githubUrl: 'https://github.com/example/recommendation-engine',
    caseStudyUrl: '/blog/case-study-ecommerce-recommendations',
    impact: 'Generated $100M+ in additional revenue',
    successStory: 'Increased average order value by 25% and repeat purchases by 40%',
    metrics: { users: '10M+', conversion: '+45%', revenue: '$100M+' },
  },
  {
    title: 'Climate Change Prediction',
    description: 'Time-series forecasting for climate patterns using RNN architecture. Predicting weather patterns 30 days ahead with 85% accuracy.',
    author: 'Prof. James Miller',
    organization: 'Climate Research Institute',
    image: '/img/showcase/climate.jpg',
    tags: ['Time Series', 'Research', 'RNN'],
    githubUrl: 'https://github.com/example/climate-prediction',
    paperUrl: 'https://arxiv.org/example',
    impact: 'Influencing climate policy in 12 countries',
    metrics: { accuracy: '85%', horizon: '30 days', coverage: 'Global' },
  },
  {
    title: 'Educational Chatbot',
    description: 'AI tutor helping students learn programming concepts. Used by 50,000+ students across 100+ universities.',
    author: 'Maria Garcia',
    organization: 'EduTech Solutions',
    image: '/img/showcase/chatbot.jpg',
    tags: ['NLP', 'Education', 'Chatbot'],
    githubUrl: 'https://github.com/example/edu-chatbot',
    demoUrl: 'https://demo.example.com',
    impact: 'Improved student grades by 20%',
    successStory: 'Students report 3x faster learning and 90% satisfaction rate',
    metrics: { students: '50K+', universities: '100+', satisfaction: '90%' },
  },
  {
    title: 'Industrial Quality Control',
    description: 'Defect detection in manufacturing with 99.5% accuracy. Inspecting 100,000+ products daily across automotive industry.',
    author: 'Robert Lee',
    organization: 'ManufacturePro',
    image: '/img/showcase/quality.jpg',
    tags: ['Computer Vision', 'Manufacturing', 'Production'],
    caseStudyUrl: '/blog/case-study-quality-control',
    impact: 'Reduced defect rate by 80%',
    successStory: 'Saved $10M annually in warranty claims and recalls',
    metrics: { accuracy: '99.5%', throughput: '100K/day', savings: '$10M/year' },
  },
  {
    title: 'Agricultural Crop Disease Detection',
    description: 'Mobile app using CNN to detect crop diseases from photos. Helping 200,000+ farmers in developing countries.',
    author: 'Dr. Priya Sharma',
    organization: 'AgriTech Foundation',
    image: '/img/showcase/agriculture.jpg',
    tags: ['Computer Vision', 'Social Impact', 'Mobile'],
    githubUrl: 'https://github.com/example/crop-disease',
    impact: 'Increased crop yields by 30%',
    successStory: 'Prevented crop losses worth $50M and improved food security',
    metrics: { farmers: '200K+', accuracy: '94%', countries: '15' },
  },
  {
    title: 'Music Generation AI',
    description: 'Transformer-based model creating original music compositions. Used by 10,000+ content creators and musicians.',
    author: 'Jessica Kim',
    organization: 'HarmonyAI',
    image: '/img/showcase/music.jpg',
    tags: ['Creative AI', 'Transformer', 'Audio'],
    githubUrl: 'https://github.com/example/music-gen',
    demoUrl: 'https://demo.example.com',
    impact: 'Generated 100,000+ unique compositions',
    metrics: { users: '10K+', compositions: '100K+', genres: '20+' },
  },
  {
    title: 'Wildlife Conservation Tracking',
    description: 'Computer vision system tracking endangered species populations. Monitoring 50+ species across 10 countries.',
    author: 'Dr. Tom Harrison',
    organization: 'Wildlife Conservation Network',
    image: '/img/showcase/wildlife.jpg',
    tags: ['Computer Vision', 'Conservation', 'Research'],
    githubUrl: 'https://github.com/example/wildlife-tracking',
    paperUrl: 'https://arxiv.org/example',
    impact: 'Protected 5,000+ endangered animals',
    successStory: 'Helped reduce poaching by 60% through early detection',
    metrics: { species: '50+', cameras: '1000+', countries: '10' },
  },
];

const allTags = [...new Set(showcaseProjects.flatMap(p => p.tags))].sort();

function ShowcaseCard({ project }) {
  return (
    <div className={`showcase-item ${project.featured ? 'showcase-item--featured' : ''}`}>
      {project.featured && (
        <div className="showcase-item__badge">⭐ Featured</div>
      )}
      <div 
        className="showcase-item__image" 
        style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: '3rem'
        }}>
        {project.title.charAt(0)}
      </div>
      <div className="showcase-item__content">
        <h3 className="showcase-item__title">{project.title}</h3>
        <p className="showcase-item__description">{project.description}</p>
        <p style={{fontSize: '0.9rem', marginBottom: '0.5rem'}}>
          <strong>{project.author}</strong> · {project.organization}
        </p>
        {project.impact && (
          <div style={{
            background: 'var(--ifm-color-success-lightest)',
            padding: '0.5rem',
            borderRadius: '4px',
            marginBottom: '0.5rem',
            fontSize: '0.85rem',
            color: 'var(--ifm-color-success-darkest)'
          }}>
            <strong>Impact:</strong> {project.impact}
          </div>
        )}
        {project.metrics && (
          <div style={{
            display: 'flex',
            gap: '0.5rem',
            marginBottom: '0.5rem',
            flexWrap: 'wrap'
          }}>
            {Object.entries(project.metrics).map(([key, value]) => (
              <span key={key} style={{
                background: 'var(--ifm-color-emphasis-200)',
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                fontSize: '0.75rem',
                fontWeight: 'bold'
              }}>
                {key}: {value}
              </span>
            ))}
          </div>
        )}
        <div className="showcase-item__tags">
          {project.tags.map((tag, idx) => (
            <span key={idx} className="showcase-item__tag">{tag}</span>
          ))}
        </div>
        <div style={{marginTop: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap'}}>
          {project.githubUrl && (
            <a href={project.githubUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          )}
          {project.demoUrl && (
            <a href={project.demoUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              Demo
            </a>
          )}
          {project.caseStudyUrl && (
            <Link to={project.caseStudyUrl} className="button button--sm button--primary">
              Case Study
            </Link>
          )}
          {project.paperUrl && (
            <a href={project.paperUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              Paper
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Showcase() {
  const [selectedTag, setSelectedTag] = useState('All');
  const [showSuccessStories, setShowSuccessStories] = useState(false);
  
  const filteredProjects = selectedTag === 'All' 
    ? showcaseProjects 
    : showcaseProjects.filter(p => p.tags.includes(selectedTag));

  const featuredProjects = showcaseProjects.filter(p => p.featured);
  const successStoryProjects = showcaseProjects.filter(p => p.successStory);

  return (
    <Layout
      title="Showcase"
      description="Discover amazing projects built with Neural DSL">
      <div className="container margin-vert--lg">
        <div style={{textAlign: 'center', marginBottom: '3rem'}}>
          <h1>Community Showcase</h1>
          <p style={{fontSize: '1.25rem', color: 'var(--ifm-color-emphasis-600)'}}>
            Discover amazing projects built with Neural DSL by our community
          </p>
          <p style={{fontSize: '1rem', color: 'var(--ifm-color-emphasis-500)', marginTop: '1rem'}}>
            Real-world applications across healthcare, finance, education, and more
          </p>
        </div>

        <div style={{marginBottom: '3rem'}}>
          <div style={{display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '1rem'}}>
            <button
              className={`button button--lg ${!showSuccessStories ? 'button--primary' : 'button--outline button--primary'}`}
              onClick={() => setShowSuccessStories(false)}>
              All Projects
            </button>
            <button
              className={`button button--lg ${showSuccessStories ? 'button--primary' : 'button--outline button--primary'}`}
              onClick={() => setShowSuccessStories(true)}>
              Success Stories
            </button>
          </div>
        </div>

        {showSuccessStories ? (
          <div>
            <div style={{textAlign: 'center', marginBottom: '3rem'}}>
              <h2>Success Stories</h2>
              <p style={{fontSize: '1.1rem', color: 'var(--ifm-color-emphasis-600)'}}>
                Real impact from our community
              </p>
            </div>
            <div className="showcase-grid">
              {successStoryProjects.map((project, idx) => (
                <div key={idx} style={{
                  background: 'var(--ifm-color-emphasis-100)',
                  padding: '2rem',
                  borderRadius: '8px',
                  border: '2px solid var(--ifm-color-primary)'
                }}>
                  <h3>{project.title}</h3>
                  <p style={{marginBottom: '1rem'}}><strong>{project.author}</strong> · {project.organization}</p>
                  <p style={{fontSize: '1.1rem', fontStyle: 'italic', marginBottom: '1rem'}}>
                    "{project.successStory}"
                  </p>
                  <div style={{
                    background: 'var(--ifm-color-success-lightest)',
                    padding: '1rem',
                    borderRadius: '4px',
                    marginTop: '1rem'
                  }}>
                    <strong>Impact:</strong> {project.impact}
                  </div>
                  {project.metrics && (
                    <div style={{
                      display: 'flex',
                      gap: '0.5rem',
                      marginTop: '1rem',
                      flexWrap: 'wrap'
                    }}>
                      {Object.entries(project.metrics).map(([key, value]) => (
                        <span key={key} style={{
                          background: 'var(--ifm-color-primary)',
                          color: 'white',
                          padding: '0.5rem 1rem',
                          borderRadius: '4px',
                          fontSize: '0.9rem',
                          fontWeight: 'bold'
                        }}>
                          {key}: {value}
                        </span>
                      ))}
                    </div>
                  )}
                  {project.caseStudyUrl && (
                    <div style={{marginTop: '1rem'}}>
                      <Link to={project.caseStudyUrl} className="button button--primary">
                        Read Full Case Study
                      </Link>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div>
            {featuredProjects.length > 0 && (
              <div style={{marginBottom: '4rem'}}>
                <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>⭐ Featured Projects</h2>
                <div className="showcase-grid">
                  {featuredProjects.map((project, idx) => (
                    <ShowcaseCard key={idx} project={project} />
                  ))}
                </div>
              </div>
            )}

            <div style={{marginBottom: '2rem', textAlign: 'center'}}>
              <h3 style={{marginBottom: '1rem'}}>Filter by Category</h3>
              <div style={{display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center'}}>
                <button
                  className={`button button--sm ${selectedTag === 'All' ? 'button--primary' : 'button--outline button--primary'}`}
                  onClick={() => setSelectedTag('All')}>
                  All ({showcaseProjects.length})
                </button>
                {allTags.map(tag => (
                  <button
                    key={tag}
                    className={`button button--sm ${selectedTag === tag ? 'button--primary' : 'button--outline button--primary'}`}
                    onClick={() => setSelectedTag(tag)}>
                    {tag} ({showcaseProjects.filter(p => p.tags.includes(tag)).length})
                  </button>
                ))}
              </div>
            </div>

            <div className="showcase-grid">
              {filteredProjects.map((project, idx) => (
                <ShowcaseCard key={idx} project={project} />
              ))}
            </div>
          </div>
        )}

        <div style={{textAlign: 'center', marginTop: '4rem', padding: '3rem', background: 'var(--ifm-color-emphasis-100)', borderRadius: '8px'}}>
          <h2>Submit Your Project</h2>
          <p style={{fontSize: '1.1rem', marginBottom: '2rem'}}>
            Built something awesome with Neural DSL? Share it with the community!
          </p>
          <a
            className="button button--primary button--lg"
            href="https://github.com/Lemniscate-world/Neural/issues/new?template=showcase.md"
            target="_blank"
            rel="noopener noreferrer">
            Submit Your Project
          </a>
        </div>
      </div>
    </Layout>
  );
}

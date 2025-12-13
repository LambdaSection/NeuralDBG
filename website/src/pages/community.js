import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const communityStats = {
  members: '5,000+',
  models: '500+',
  downloads: '50,000+',
  universities: '100+'
};

const events = [
  {
    title: 'Neural DSL Workshop: Advanced Architectures',
    date: 'February 20, 2024',
    time: '2:00 PM EST',
    type: 'Workshop',
    host: 'Dr. Sarah Chen',
    description: 'Deep dive into advanced neural network architectures including Transformers and GANs.',
    registrationUrl: 'https://discord.gg/KFku4KvS'
  },
  {
    title: 'Monthly Community Meetup',
    date: 'February 28, 2024',
    time: '6:00 PM EST',
    type: 'Meetup',
    host: 'Neural DSL Team',
    description: 'Share your projects, ask questions, and network with the community.',
    registrationUrl: 'https://discord.gg/KFku4KvS'
  },
  {
    title: 'Model Optimization Hackathon',
    date: 'March 15-17, 2024',
    time: '48-hour event',
    type: 'Hackathon',
    host: 'Neural DSL Community',
    description: 'Compete to build the most efficient model. Prizes for top 3 teams!',
    registrationUrl: 'https://discord.gg/KFku4KvS'
  }
];

const topContributors = [
  { username: 'alice_ai', points: 1250, badges: ['Legend', 'Prolific Creator'], models: 15 },
  { username: 'bob_ml', points: 890, badges: ['Community Champion', 'Model Reviewer'], models: 8 },
  { username: 'charlie_nn', points: 720, badges: ['Rising Star', 'Community Helper'], models: 6 },
  { username: 'diana_dl', points: 650, badges: ['Model Publisher', 'Community Helper'], models: 5 },
  { username: 'evan_cv', points: 580, badges: ['Rising Star', 'Model Reviewer'], models: 4 }
];

const badges = [
  { name: 'First Model', icon: '🎯', description: 'Upload your first model', requirement: '1 model' },
  { name: 'Model Publisher', icon: '📦', description: 'Published 5 models', requirement: '5 models' },
  { name: 'Prolific Creator', icon: '⭐', description: 'Published 10+ models', requirement: '10 models' },
  { name: 'Model Reviewer', icon: '📝', description: 'Rated 10+ models', requirement: '10 ratings' },
  { name: 'Community Helper', icon: '🤝', description: 'Helped others 20+ times', requirement: '20 helpful comments' },
  { name: 'Rising Star', icon: '🌟', description: 'Earned 100+ points', requirement: '100 points' },
  { name: 'Community Champion', icon: '🏆', description: 'Earned 500+ points', requirement: '500 points' },
  { name: 'Legend', icon: '👑', description: 'Earned 1000+ points', requirement: '1000 points' }
];

const resources = [
  {
    title: 'Getting Started Guide',
    description: 'Complete beginner-friendly guide to Neural DSL',
    link: '/docs/intro',
    icon: '📚'
  },
  {
    title: 'Tutorial Library',
    description: '50+ tutorials covering basic to advanced topics',
    link: '/docs/tutorial',
    icon: '🎓'
  },
  {
    title: 'Course Materials',
    description: 'University-quality courses and assignments',
    link: '/docs/courses',
    icon: '🏫'
  },
  {
    title: 'Example Projects',
    description: 'Real-world examples you can learn from',
    link: '/showcase',
    icon: '💡'
  }
];

export default function Community() {
  return (
    <Layout
      title="Community"
      description="Join the Neural DSL community and connect with developers worldwide">
      <div className="container margin-vert--lg">
        
        {/* Header */}
        <div style={{textAlign: 'center', marginBottom: '4rem'}}>
          <h1 style={{fontSize: '3rem', marginBottom: '1rem'}}>Join Our Community</h1>
          <p style={{fontSize: '1.3rem', color: 'var(--ifm-color-emphasis-600)', marginBottom: '2rem'}}>
            Connect with developers, share knowledge, and build amazing neural networks together
          </p>
          <div style={{display: 'flex', gap: '1rem', justifyContent: 'center'}}>
            <a
              className="button button--primary button--lg"
              href="https://discord.gg/KFku4KvS"
              target="_blank"
              rel="noopener noreferrer">
              Join Discord Server
            </a>
            <Link
              className="button button--outline button--primary button--lg"
              to="/showcase">
              View Showcase
            </Link>
          </div>
        </div>

        {/* Stats */}
        <div style={{marginBottom: '4rem'}}>
          <div className="row">
            {Object.entries(communityStats).map(([key, value]) => (
              <div key={key} className="col col--3">
                <div style={{
                  textAlign: 'center',
                  padding: '2rem',
                  background: 'var(--ifm-color-emphasis-100)',
                  borderRadius: '8px'
                }}>
                  <div style={{fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--ifm-color-primary)'}}>
                    {value}
                  </div>
                  <div style={{fontSize: '1.1rem', textTransform: 'capitalize', marginTop: '0.5rem'}}>
                    {key}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Upcoming Events */}
        <div style={{marginBottom: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Upcoming Events</h2>
          <div className="row">
            {events.map((event, idx) => (
              <div key={idx} className="col col--4">
                <div style={{
                  padding: '2rem',
                  background: 'var(--ifm-color-emphasis-100)',
                  borderRadius: '8px',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column'
                }}>
                  <div style={{
                    display: 'inline-block',
                    padding: '0.25rem 0.75rem',
                    background: 'var(--ifm-color-primary)',
                    color: 'white',
                    borderRadius: '4px',
                    fontSize: '0.85rem',
                    marginBottom: '1rem',
                    width: 'fit-content'
                  }}>
                    {event.type}
                  </div>
                  <h3>{event.title}</h3>
                  <p style={{color: 'var(--ifm-color-emphasis-700)', marginBottom: '0.5rem'}}>
                    📅 {event.date}<br/>
                    🕐 {event.time}
                  </p>
                  <p style={{fontSize: '0.95rem', marginBottom: '1rem'}}>
                    <strong>Host:</strong> {event.host}
                  </p>
                  <p style={{flex: 1}}>{event.description}</p>
                  <a
                    href={event.registrationUrl}
                    className="button button--primary button--sm"
                    target="_blank"
                    rel="noopener noreferrer">
                    Register Now
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Leaderboard */}
        <div style={{marginBottom: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>🏆 Community Leaderboard</h2>
          <div style={{maxWidth: '800px', margin: '0 auto'}}>
            {topContributors.map((contributor, idx) => (
              <div key={idx} style={{
                display: 'flex',
                alignItems: 'center',
                padding: '1.5rem',
                marginBottom: '1rem',
                background: idx === 0 ? 'linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)' : 
                           idx === 1 ? 'linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%)' :
                           idx === 2 ? 'linear-gradient(135deg, #cd7f32 0%, #e59b5f 100%)' :
                           'var(--ifm-color-emphasis-100)',
                borderRadius: '8px',
                border: idx < 3 ? '2px solid' : 'none',
                borderColor: idx === 0 ? '#ffd700' : idx === 1 ? '#c0c0c0' : '#cd7f32'
              }}>
                <div style={{
                  fontSize: '2rem',
                  fontWeight: 'bold',
                  marginRight: '1.5rem',
                  minWidth: '50px',
                  textAlign: 'center'
                }}>
                  {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx + 1}`}
                </div>
                <div style={{flex: 1}}>
                  <div style={{fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.25rem'}}>
                    {contributor.username}
                  </div>
                  <div style={{fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-700)'}}>
                    {contributor.points} points • {contributor.models} models
                  </div>
                  <div style={{marginTop: '0.5rem'}}>
                    {contributor.badges.map((badge, bidx) => (
                      <span key={bidx} style={{
                        display: 'inline-block',
                        padding: '0.25rem 0.5rem',
                        background: 'var(--ifm-color-primary-lightest)',
                        color: 'var(--ifm-color-primary-darkest)',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        marginRight: '0.5rem'
                      }}>
                        {badge}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div style={{textAlign: 'center', marginTop: '2rem'}}>
            <Link to="/marketplace" className="button button--primary">
              View Full Leaderboard
            </Link>
          </div>
        </div>

        {/* Badges */}
        <div style={{marginBottom: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Achievement Badges</h2>
          <p style={{textAlign: 'center', marginBottom: '2rem', color: 'var(--ifm-color-emphasis-600)'}}>
            Earn badges by contributing to the community
          </p>
          <div className="row">
            {badges.map((badge, idx) => (
              <div key={idx} className="col col--3">
                <div style={{
                  textAlign: 'center',
                  padding: '1.5rem',
                  background: 'var(--ifm-color-emphasis-100)',
                  borderRadius: '8px',
                  marginBottom: '1rem'
                }}>
                  <div style={{fontSize: '3rem', marginBottom: '0.5rem'}}>{badge.icon}</div>
                  <div style={{fontWeight: 'bold', marginBottom: '0.5rem'}}>{badge.name}</div>
                  <div style={{fontSize: '0.85rem', color: 'var(--ifm-color-emphasis-700)', marginBottom: '0.5rem'}}>
                    {badge.description}
                  </div>
                  <div style={{fontSize: '0.75rem', color: 'var(--ifm-color-emphasis-600)'}}>
                    {badge.requirement}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Learning Resources */}
        <div style={{marginBottom: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Learning Resources</h2>
          <div className="row">
            {resources.map((resource, idx) => (
              <div key={idx} className="col col--3">
                <Link to={resource.link} style={{textDecoration: 'none', color: 'inherit'}}>
                  <div style={{
                    textAlign: 'center',
                    padding: '2rem',
                    background: 'var(--ifm-color-emphasis-100)',
                    borderRadius: '8px',
                    height: '100%',
                    transition: 'transform 0.2s',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                    <div style={{fontSize: '3rem', marginBottom: '1rem'}}>{resource.icon}</div>
                    <h3>{resource.title}</h3>
                    <p>{resource.description}</p>
                  </div>
                </Link>
              </div>
            ))}
          </div>
        </div>

        {/* Ways to Contribute */}
        <div style={{marginBottom: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Ways to Contribute</h2>
          <div className="row">
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>📦</div>
                <h3>Share Models</h3>
                <p>Upload your trained models to help others get started quickly</p>
              </div>
            </div>
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>💬</div>
                <h3>Help Others</h3>
                <p>Answer questions on Discord and build your reputation</p>
              </div>
            </div>
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>📝</div>
                <h3>Write Tutorials</h3>
                <p>Create learning resources to help newcomers</p>
              </div>
            </div>
          </div>
          <div className="row">
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>🐛</div>
                <h3>Report Issues</h3>
                <p>Help us improve by reporting bugs and suggesting features</p>
              </div>
            </div>
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>⭐</div>
                <h3>Review Models</h3>
                <p>Rate and review models to guide others</p>
              </div>
            </div>
            <div className="col col--4">
              <div style={{padding: '2rem', textAlign: 'center'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>📢</div>
                <h3>Spread the Word</h3>
                <p>Share Neural DSL with your network</p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div style={{
          textAlign: 'center',
          padding: '4rem 2rem',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '12px',
          color: 'white'
        }}>
          <h2 style={{color: 'white', marginBottom: '1rem'}}>Ready to Join?</h2>
          <p style={{fontSize: '1.2rem', marginBottom: '2rem', opacity: 0.9}}>
            Connect with thousands of developers building the future of neural networks
          </p>
          <div style={{display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap'}}>
            <a
              className="button button--secondary button--lg"
              href="https://discord.gg/KFku4KvS"
              target="_blank"
              rel="noopener noreferrer">
              Join Discord
            </a>
            <a
              className="button button--outline button--secondary button--lg"
              href="https://github.com/Lemniscate-world/Neural"
              target="_blank"
              rel="noopener noreferrer"
              style={{borderColor: 'white', color: 'white'}}>
              Star on GitHub
            </a>
          </div>
        </div>

      </div>
    </Layout>
  );
}

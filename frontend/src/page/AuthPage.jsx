import { useEffect, useState } from 'react'
import { t } from '../i18n'
import './AuthPage.css'

export default function AuthPage({ lang, onBack, onRegister, onLogin, defaultRole = 'passenger' }) {
  const [tab, setTab] = useState('login')

  const [role, setRole] = useState(defaultRole === 'driver' ? 'driver' : 'passenger')

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [carType, setCarType] = useState('')

  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (defaultRole === 'driver') setRole('driver')
    else if (defaultRole === 'passenger') setRole('passenger')
  }, [defaultRole])

  const mapLoginErrorToKey = msg => {
    const s = String(msg || '').trim()

    if (s === 'invalid credentials') return 'errorWrongPassword'
    if (s === 'not a passenger account') return 'errorNotPassengerAccount'
    if (s === 'not a driver account') return 'errorNotDriverAccount'

    if (s === 'auth_loginFailed') return 'errorWrongPassword'
    if (s === 'loginFailed') return 'errorUserNotFound'

    return s || 'loginFailed'
  }

  const mapRegisterErrorToKey = msg => {
    const s = String(msg || '').trim()
    if (s === 'username already exists') return 'errorUsernameTaken'
    if (s === 'auth_registerFailed') return 'errorUsernameTaken'
    return s || 'registerFailed'
  }

  const handleSubmit = async e => {
    e.preventDefault()
    if (submitting) return
    setError('')
    setSubmitting(true)

    try {
      if (tab === 'login') {
        const result = (await onLogin?.({ username, password, role })) || { ok: false }

        if (!result.ok) {
          setError(mapLoginErrorToKey(result.message))
        }
        return
      }

      if (!username || !password) {
        setError('errorMissingFields')
        return
      }
      if (role === 'driver' && !carType) {
        setError('selectCarTypeHint')
        return
      }

      const payload = {
        username,
        password,
        role,
        carType: role === 'driver' ? carType : null,
      }

      const result = (await onRegister?.(payload)) || { ok: false }

      if (!result.ok) {
        setError(mapRegisterErrorToKey(result.message))
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="auth-page-root">
      <div className="auth-card">
        <div className="auth-header">
          <h2 className="auth-title">{t(lang, 'authTitle')}</h2>
          {onBack && (
            <button type="button" className="auth-back-btn" onClick={onBack}>
              {t(lang, 'backHome')}
            </button>
          )}
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={'auth-tab' + (tab === 'login' ? ' auth-tab-active' : '')}
            onClick={() => {
              setTab('login')
              setError('')
            }}
          >
            {t(lang, 'login')}
          </button>
          <button
            type="button"
            className={'auth-tab' + (tab === 'register' ? ' auth-tab-active' : '')}
            onClick={() => {
              setTab('register')
              setError('')
            }}
          >
            {t(lang, 'register')}
          </button>
        </div>

        {error && <div className="auth-error-banner">{t(lang, error)}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-field">
            <label className="auth-label">{t(lang, 'roleLabel')}</label>
            <div className="auth-radio-group">
              <label>
                <input
                  type="radio"
                  value="passenger"
                  checked={role === 'passenger'}
                  onChange={() => {
                    setRole('passenger')
                    setCarType('')
                  }}
                />
                {t(lang, 'rolePassenger')}
              </label>
              <label>
                <input
                  type="radio"
                  value="driver"
                  checked={role === 'driver'}
                  onChange={() => setRole('driver')}
                />
                {t(lang, 'roleDriver')}
              </label>
            </div>
          </div>

          {tab === 'register' && role === 'driver' && (
            <div className="auth-field">
              <label className="auth-label">{t(lang, 'carTypeLabel')}</label>
              <select className="auth-input" value={carType} onChange={e => setCarType(e.target.value)}>
                <option value="">{t(lang, 'selectCarTypeHint')}</option>
                <option value="YELLOW">{t(lang, 'carTypeYellow')}</option>
                <option value="GREEN">{t(lang, 'carTypeGreen')}</option>
                <option value="FHV">{t(lang, 'carTypeFhv')}</option>
              </select>
            </div>
          )}

          <div className="auth-field">
            <label className="auth-label">{t(lang, 'usernameLabel')}</label>
            <input
              className="auth-input"
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder={t(lang, 'usernamePlaceholder')}
            />
          </div>

          <div className="auth-field">
            <label className="auth-label">{t(lang, 'passwordLabel')}</label>
            <input
              className="auth-input"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder={t(lang, 'passwordPlaceholder')}
            />
          </div>

          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {tab === 'login' ? t(lang, 'login') : t(lang, 'registerNow')}
          </button>
        </form>
      </div>
    </div>
  )
}

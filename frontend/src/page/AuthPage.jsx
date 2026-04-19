// src/views/AuthPage.jsx
import { useEffect, useState } from 'react'
import { t } from '../i18n'
import './AuthPage.css'

export default function AuthPage({ lang, onBack, onRegister, onLogin, defaultRole = 'passenger' }) {
  const [tab, setTab] = useState('login') // 'login' | 'register'

  // ✅ role：登入/註冊都使用（登入要決定打哪個 endpoint）
  const [role, setRole] = useState(defaultRole === 'driver' ? 'driver' : 'passenger')

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [carType, setCarType] = useState('') // driver 註冊用

  const [error, setError] = useState('') // 存「錯誤 key」
  const [submitting, setSubmitting] = useState(false)

  // ✅ 如果 App 依 URL 推了 defaultRole（例如 /?role=driver&auth=1），同步到表單
  useEffect(() => {
    if (defaultRole === 'driver') setRole('driver')
    else if (defaultRole === 'passenger') setRole('passenger')
  }, [defaultRole])

  const mapLoginErrorToKey = msg => {
    const s = String(msg || '').trim()

    // 後端原生字串（你的 FastAPI）
    if (s === 'invalid credentials') return 'errorWrongPassword' // 或你想用 errorInvalidCredentials
    if (s === 'not a passenger account') return 'errorNotPassengerAccount'
    if (s === 'not a driver account') return 'errorNotDriverAccount'

    // 兼容你舊的 key/文字
    if (s === 'auth_loginFailed') return 'errorWrongPassword'
    if (s === 'loginFailed') return 'errorUserNotFound'

    // 若後端直接回我們的 key（或你自己丟回來）
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
        // ✅ 登入：把 role 一起送回 App，讓 App 不用猜 endpoint
        const result = (await onLogin?.({ username, password, role })) || { ok: false }

        if (!result.ok) {
          setError(mapLoginErrorToKey(result.message))
        }
        return
      }

      // ===== 註冊 =====
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
        {/* 標題列 */}
        <div className="auth-header">
          <h2 className="auth-title">{t(lang, 'authTitle')}</h2>
          {onBack && (
            <button type="button" className="auth-back-btn" onClick={onBack}>
              {t(lang, 'backHome')}
            </button>
          )}
        </div>

        {/* 登入 / 註冊 tab */}
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

        {/* 錯誤訊息（key → i18n） */}
        {error && <div className="auth-error-banner">{t(lang, error)}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          {/* ✅ 身分：登入/註冊都顯示，避免 App 猜錯 endpoint */}
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

          {/* ✅ 只有「註冊 + 司機」才需要車種 */}
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

          {/* 帳號 */}
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

          {/* 密碼 */}
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

// src/LandingPage.jsx
import { useEffect, useState } from 'react'
import './LandingPage.css'
import { t, languages } from './i18n'

export default function LandingPage({
  lang,
  onChangeLang,
  onPassengerClick,
  onDriverClick,
  onAuthClick,
}) {
  // ====== Landing 輸入：上下車 + 建議 + 座標 ======
  const [pickupText, setPickupText] = useState('')
  const [dropoffText, setDropoffText] = useState('')

  const [pickupSuggestions, setPickupSuggestions] = useState([])
  const [dropoffSuggestions, setDropoffSuggestions] = useState([])

  const [pickupLoc, setPickupLoc] = useState(null)
  const [dropoffLoc, setDropoffLoc] = useState(null)

  const [pickupLocked, setPickupLocked] = useState(false)
  const [dropoffLocked, setDropoffLocked] = useState(false)

  // 上車建議
  useEffect(() => {
    if (pickupLocked) {
      setPickupSuggestions([])
      return
    }
    if (!pickupText || pickupText.trim().length < 2) {
      setPickupSuggestions([])
      return
    }

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/geocode?q=${encodeURIComponent(pickupText.trim())}`, {
          signal: controller.signal,
        })
        if (!res.ok) return
        const data = await res.json()
        setPickupSuggestions(Array.isArray(data) ? data : [])
      } catch {
        // ignore
      }
    }, 400)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [pickupText, pickupLocked])

  // 下車建議
  useEffect(() => {
    if (dropoffLocked) {
      setDropoffSuggestions([])
      return
    }
    if (!dropoffText || dropoffText.trim().length < 2) {
      setDropoffSuggestions([])
      return
    }

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/geocode?q=${encodeURIComponent(dropoffText.trim())}`, {
          signal: controller.signal,
        })
        if (!res.ok) return
        const data = await res.json()
        setDropoffSuggestions(Array.isArray(data) ? data : [])
      } catch {
        // ignore
      }
    }, 400)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [dropoffText, dropoffLocked])

  const handleSelectPickup = item => {
    setPickupText(item.label)
    setPickupLoc({ lat: item.lat, lng: item.lng })
    setPickupSuggestions([])
    setPickupLocked(true)
  }

  const handleSelectDropoff = item => {
    setDropoffText(item.label)
    setDropoffLoc({ lat: item.lat, lng: item.lng })
    setDropoffSuggestions([])
    setDropoffLocked(true)
  }

  const goPassengerWithDraft = () => {
    onPassengerClick?.({
      pickupText: pickupText.trim(),
      dropoffText: dropoffText.trim(),
      pickupLoc,
      dropoffLoc,
      stops: [],
    })
  }

  return (
    <div id="top" className="landing-root">
      {/* 導覽列 */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div className="container">
          <a className="navbar-brand fw-bold" href="#top">
            SmartDispatch
          </a>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
          >
            <span className="navbar-toggler-icon" />
          </button>

          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto">
              <li className="nav-item">
                <a
                  className="nav-link active"
                  href="#"
                  onClick={e => {
                    e.preventDefault()
                    goPassengerWithDraft()
                  }}
                >
                  {t(lang, 'landingNavPassenger')}
                </a>
              </li>

              <li className="nav-item">
                <a
                  className="nav-link"
                  href="#"
                  onClick={e => {
                    e.preventDefault()
                    onDriverClick?.()
                  }}
                >
                  {t(lang, 'landingNavDriver')}
                </a>
              </li>

              <li className="nav-item">
                <a
                  className="nav-link btn btn-primary text-white ms-2 px-3"
                  href="#"
                  onClick={e => {
                    e.preventDefault()
                    onAuthClick?.()
                  }}
                >
                  {t(lang, 'landingNavAuth')}
                </a>
              </li>
            </ul>

            {/* 語言切換 */}
            <div className="ms-3 d-flex align-items-center text-white">
              <span className="small me-2">{t(lang, 'language')}：</span>
              <select
                className="form-select form-select-sm bg-dark text-white border-secondary"
                style={{ width: 90 }}
                value={lang}
                onChange={e => onChangeLang?.(e.target.value)}
              >
                {languages.map(code => (
                  <option key={code} value={code}>
                    {code.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </nav>

      {/* 乘客 hero 區塊 */}
      <section className="hero-section" id="passenger">
        <div className="hero-overlay" />
        <div className="container hero-content">
          <div className="row align-items-center">
            <div className="col-lg-6 text-white mb-5 mb-lg-0">
              <h1 className="display-4 fw-bold">{t(lang, 'landingHeroTitle')}</h1>
              <p className="lead mb-4">{t(lang, 'landingHeroSubtitle')}</p>
            </div>

            <div className="col-lg-5 offset-lg-1">
              <div className="booking-card">
                <h3 className="fw-bold mb-4">{t(lang, 'landingHeroWhereTo')}</h3>

                <form>
                  {/* 上車 */}
                  <div className="mb-3 position-relative">
                    <label className="form-label text-muted small">
                      {t(lang, 'landingHeroPickupLabel')}
                    </label>
                    <input
                      type="text"
                      className="form-control form-control-lg"
                      placeholder="輸入上車位置"
                      value={pickupText}
                      onChange={e => {
                        setPickupText(e.target.value)
                        setPickupLoc(null)
                        setPickupLocked(false)
                      }}
                    />

                    {pickupSuggestions.length > 0 && (
                      <div
                        className="autocomplete-dropdown"
                        style={{
                          position: 'absolute',
                          zIndex: 10,
                          left: 0,
                          right: 0,
                          top: '100%',
                          marginTop: 6,
                          background: '#fff',
                          border: '1px solid #e9ecef',
                          borderRadius: 8,
                          overflow: 'hidden',
                        }}
                      >
                        {pickupSuggestions.map((item, idx) => (
                          <button
                            key={idx}
                            type="button"
                            className="autocomplete-item"
                            style={{
                              width: '100%',
                              textAlign: 'left',
                              padding: '10px 12px',
                              border: 'none',
                              background: 'transparent',
                              cursor: 'pointer',
                            }}
                            onClick={() => handleSelectPickup(item)}
                          >
                            {item.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* 下車 */}
                  <div className="mb-3 position-relative">
                    <label className="form-label text-muted small">
                      {t(lang, 'landingHeroDropoffLabel')}
                    </label>
                    <input
                      type="text"
                      className="form-control form-control-lg"
                      placeholder={t(lang, 'landingHeroDropoffPlaceholder')}
                      value={dropoffText}
                      onChange={e => {
                        setDropoffText(e.target.value)
                        setDropoffLoc(null)
                        setDropoffLocked(false)
                      }}
                    />

                    {dropoffSuggestions.length > 0 && (
                      <div
                        className="autocomplete-dropdown"
                        style={{
                          position: 'absolute',
                          zIndex: 10,
                          left: 0,
                          right: 0,
                          top: '100%',
                          marginTop: 6,
                          background: '#fff',
                          border: '1px solid #e9ecef',
                          borderRadius: 8,
                          overflow: 'hidden',
                        }}
                      >
                        {dropoffSuggestions.map((item, idx) => (
                          <button
                            key={idx}
                            type="button"
                            className="autocomplete-item"
                            style={{
                              width: '100%',
                              textAlign: 'left',
                              padding: '10px 12px',
                              border: 'none',
                              background: 'transparent',
                              cursor: 'pointer',
                            }}
                            onClick={() => handleSelectDropoff(item)}
                          >
                            {item.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* 查看價格與車輛 → 直接進入乘客端（並帶草稿） */}
                  <button
                    type="button"
                    className="btn btn-dark w-100 btn-lg py-3 fw-bold"
                    onClick={goPassengerWithDraft}
                  >
                    {t(lang, 'landingHeroCta')}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 以下原樣保留（你的流程圖、司機招募、footer 不動） */}
      <section className="how-it-works-section py-5 bg-white">
        <div className="container text-center">
          <div className="mb-5">
            <span className="text-warning fw-bold text-uppercase ls-1">
              {t(lang, 'landingHowTitleTag')}
            </span>
            <h2 className="fw-bold mt-2">{t(lang, 'landingHowTitle')}</h2>
            <p className="text-muted">{t(lang, 'landingHowSubtitle')}</p>
          </div>

          <div className="row justify-content-center">
            <div className="col-md-4 mb-4 mb-md-0 position-relative">
              <div className="step-card p-4">
                <div className="icon-circle bg-warning text-dark mb-4 mx-auto d-flex align-items-center justify-content-center shadow">
                  <i className="bi bi-geo-alt-fill fs-2" />
                </div>
                <h4 className="fw-bold">{t(lang, 'landingHowStep1Title')}</h4>
                <p className="text-muted">{t(lang, 'landingHowStep1Desc')}</p>
              </div>
              <div className="d-none d-md-block position-absolute top-50 end-0 translate-middle-y text-muted">
                <i className="bi bi-chevron-right fs-1" />
              </div>
            </div>

            <div className="col-md-4 mb-4 mb-md-0 position-relative">
              <div className="step-card p-4">
                <div className="icon-circle bg-dark text-warning mb-4 mx-auto d-flex align-items-center justify-content-center shadow">
                  <i className="bi bi-cpu-fill fs-2" />
                </div>
                <h4 className="fw-bold">{t(lang, 'landingHowStep2Title')}</h4>
                <p className="text-muted">{t(lang, 'landingHowStep2Desc')}</p>
              </div>
              <div className="d-none d-md-block position-absolute top-50 end-0 translate-middle-y text-muted">
                <i className="bi bi-chevron-right fs-1" />
              </div>
            </div>

            <div className="col-md-4">
              <div className="step-card p-4">
                <div className="icon-circle bg-warning text-dark mb-4 mx-auto d-flex align-items-center justify-content-center shadow">
                  <i className="bi bi-emoji-smile-fill fs-2" />
                </div>
                <h4 className="fw-bold">{t(lang, 'landingHowStep3Title')}</h4>
                <p className="text-muted">{t(lang, 'landingHowStep3Desc')}</p>
              </div>
            </div>
          </div>

          <div className="mt-5">
            <button
              type="button"
              className="btn btn-dark btn-lg px-5 rounded-pill shadow-sm"
              onClick={goPassengerWithDraft}
            >
              {t(lang, 'landingHowCta')}
              <i className="bi bi-arrow-right ms-2" />
            </button>
          </div>
        </div>
      </section>

      <section className="driver-section" id="driver">
        <div className="container">
          <div className="row align-items-center">
            <div className="col-md-6 mb-4 mb-md-0">
              <img
                src="https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?q=80&w=800&auto=format&fit=crop"
                className="img-fluid rounded shadow"
                alt="Driver App UI"
              />
            </div>
            <div className="col-md-6">
              <span className="badge bg-warning text-dark mb-2">
                {t(lang, 'landingDriverBadge')}
              </span>
              <h2 className="fw-bold mb-3">
                {t(lang, 'landingDriverTitleLine1')}
                <br />
                {t(lang, 'landingDriverTitleLine2')}
              </h2>
              <p className="text-muted">{t(lang, 'landingDriverIntro')}</p>

              <ul className="list-unstyled mt-4">
                <li className="mb-3">
                  <h5 className="fw-bold">{t(lang, 'landingDriverFeature1Title')}</h5>
                  <p className="small text-muted">{t(lang, 'landingDriverFeature1Desc')}</p>
                </li>
                <li className="mb-3">
                  <h5 className="fw-bold">{t(lang, 'landingDriverFeature2Title')}</h5>
                  <p className="small text-muted">{t(lang, 'landingDriverFeature2Desc')}</p>
                </li>
              </ul>

              <button
                type="button"
                className="btn btn-outline-dark mt-3"
                onClick={() => onDriverClick?.()}
              >
                {t(lang, 'landingDriverCta')}
              </button>
            </div>
          </div>
        </div>
      </section>

      <footer className="bg-dark text-white py-5 text-center">
        <div className="container">
          <h3 className="mb-4">{t(lang, 'landingFooterTitle')}</h3>
          <div className="d-flex justify-content-center gap-3">
            <button className="btn btn-light btn-lg">{t(lang, 'landingFooterIos')}</button>
            <button className="btn btn-light btn-lg">{t(lang, 'landingFooterAndroid')}</button>
          </div>
          <p className="mt-5 text-white-50 small">{t(lang, 'landingFooterCopyright')}</p>
        </div>
      </footer>
    </div>
  )
}

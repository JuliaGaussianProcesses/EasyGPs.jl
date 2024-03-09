using ParameterHandling: ParameterHandling
import ParameterHandling: flatten, value

struct RealParameter{T<:Real,Tp} <: Real
    par::Tp
end

value(p::RealParameter) = value(p.par)
maybe_value(x) = value(x)
maybe_value(p::RealParameter) = p

function flatten(::Type{T1}, p::RealParameter{T2}) where {T1<:Real,T2<:Real}
    v, unflatten = flatten(T1, p.par)
    function unflatten_to_RealParameter(v::Vector{T1})
        return RealParameter{T2,typeof(p.par)}(unflatten(v))
    end
    return v, unflatten_to_RealParameter
end

Base.convert(::Type{<:RealParameter}, x::RealParameter) = x
Base.convert(::Type{T}, x::RealParameter) where {T<:Real} = convert(T, value(x))
Base.promote(x::RealParameter, y::Number) = promote(value(x), y)
Base.promote(x::Number, y::RealParameter) = reverse(promote(y, x))
Base.zero(::Type{<:RealParameter{T}}) where {T} = zero(T)
function Base.inv(p::RealParameter{T,<:ParameterHandling.Fixed}) where {T<:Real}
    return fixed(inv(value(p)))
end

function positive(x::Real)
    _x = ParameterHandling.positive(x)
    return RealParameter{typeof(x),typeof(_x)}(_x)
end

positive(p::RealParameter) = p

function fixed(x::Real)
    _x = ParameterHandling.fixed(x)
    return RealParameter{typeof(x),typeof(_x)}(_x)
end
